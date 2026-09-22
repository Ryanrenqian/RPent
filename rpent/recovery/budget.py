# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded execution accounting used to terminate recovery loops.

The figures below are measured from the terminal state of the 2026-09-20
100-cell main table (run ``rpent_gpt55_xhigh_libero_10_task_100cell``,
``results.json`` updated 2026-09-22 07:09 after pass-2 completion): ``turns_used``
has n=100, median=53, max=164, with 16 cells over 100 and 18 cells at or over
100; ``elapsed_s`` has median=5001 and max=5006, with 63 cells at or over
5000; and the median elapsed-seconds-per-turn is 51.  The elapsed-time
distribution is 7 cells below 1000 s, 27 from 1000--4000 s, 3 from
4000--4999 s, and 63 at or above 5000 s.  Both the turn ceiling and the
wall-clock ceiling are live constraints; neither may be treated as inactive.

Recovery budgets are a named fraction of the protocol cell budget.  The
default fraction is the user's 2026-09-22 decision, not an empirical estimate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import floor, isfinite
from typing import Hashable

PLANNER_TIMEOUT_S = 5000.0


@dataclass(frozen=True, slots=True)
class CellBudget:
    """The four per-cell protocol ceilings written by ``launch_task100.sh``."""

    wall_clock_s: float = PLANNER_TIMEOUT_S
    turns: int = 100
    env_steps: int = 10000
    upper_model_calls: int = 100

    def __post_init__(self) -> None:
        """Validate that every cell ceiling is positive and finite."""
        if (
            isinstance(self.wall_clock_s, bool)
            or not isinstance(self.wall_clock_s, (int, float))
            or not isfinite(self.wall_clock_s)
            or self.wall_clock_s <= 0
        ):
            raise ValueError("cell wall_clock_s must be a positive finite number")
        for name in ("turns", "env_steps", "upper_model_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"cell {name} must be a positive integer")


DEFAULT_RECOVERY_FRACTION = 0.2
"""User decision on 2026-09-22; this is not an empirical estimate."""

_DEFAULT_CELL_BUDGET = CellBudget()
DEFAULT_MAX_WALL_CLOCK_S = _DEFAULT_CELL_BUDGET.wall_clock_s * DEFAULT_RECOVERY_FRACTION
DEFAULT_MAX_TURNS = max(
    1, floor(_DEFAULT_CELL_BUDGET.turns * DEFAULT_RECOVERY_FRACTION)
)
DEFAULT_MAX_ENV_STEPS = max(
    1, floor(_DEFAULT_CELL_BUDGET.env_steps * DEFAULT_RECOVERY_FRACTION)
)
DEFAULT_MAX_UPPER_MODEL_CALLS = max(
    1, floor(_DEFAULT_CELL_BUDGET.upper_model_calls * DEFAULT_RECOVERY_FRACTION)
)


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Protocol-aligned ceilings; wall clock is deliberately below planner timeout."""

    max_turns: int = DEFAULT_MAX_TURNS
    max_wall_clock_s: float = DEFAULT_MAX_WALL_CLOCK_S
    max_upper_model_calls: int = DEFAULT_MAX_UPPER_MODEL_CALLS
    max_env_steps: int = DEFAULT_MAX_ENV_STEPS

    @classmethod
    def from_cell_budget(
        cls, cell: CellBudget, fraction: float = DEFAULT_RECOVERY_FRACTION
    ) -> "BudgetLimits":
        """Scale every recovery dimension by one cell-budget fraction.

        Args:
            cell: Protocol-level cell budget to scale.
            fraction: Fraction strictly between zero and one.

        Returns:
            Recovery limits with integer dimensions floored to at least one.

        Raises:
            TypeError: If ``cell`` is not a :class:`CellBudget`.
            ValueError: If ``fraction`` is not finite or lies outside ``(0, 1)``.
        """
        if not isinstance(cell, CellBudget):
            raise TypeError("cell must be a CellBudget")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not isfinite(fraction)
        ):
            raise ValueError(
                "fraction must be a finite number in (0, 1); "
                "recovery wall clock must be strictly below planner timeout"
            )
        if not 0 < fraction < 1:
            raise ValueError(
                "fraction must be in (0, 1); "
                "recovery wall clock must be strictly below planner timeout"
            )
        return cls(
            max_turns=max(1, floor(cell.turns * fraction)),
            max_wall_clock_s=cell.wall_clock_s * fraction,
            max_upper_model_calls=max(1, floor(cell.upper_model_calls * fraction)),
            max_env_steps=max(1, floor(cell.env_steps * fraction)),
        )

    def __post_init__(self) -> None:
        """Validate the configured recovery ceilings."""
        self._validate()

    def _validate(self) -> None:
        if self.max_wall_clock_s >= PLANNER_TIMEOUT_S:
            raise ValueError("max_wall_clock_s must be strictly below planner timeout")
        if (
            self.max_turns < 0
            or self.max_upper_model_calls < 0
            or self.max_env_steps < 0
        ):
            raise ValueError("budget ceilings must be non-negative")
        if self.max_wall_clock_s < 0:
            raise ValueError("max_wall_clock_s must be non-negative")


@dataclass(slots=True)
class BudgetLedger:
    """Track costs and terminate repeated signatures.

    A signature gives up after ``max_no_progress + 1`` occurrences within the
    most recent ``2 * (max_no_progress + 1)`` attempts.

    With the defaults, no-progress termination precedes the per-parameter
    adaptation cap.  That cap is a redundant guard by default and becomes the
    binding constraint only when max_parameter_adaptations < max_no_progress.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    max_no_progress: int = 3
    max_parameter_adaptations: int = 3
    attempts: int = 0
    turns: int = 0
    wall_clock_s: float = 0.0
    upper_model_calls: int = 0
    env_steps: int = 0
    _recent_signatures: deque[Hashable | None] = field(init=False, repr=False)
    _no_progress_count: int = field(default=0, init=False, repr=False)
    termination_reason: str | None = field(default=None, init=False)
    parameter_adaptations: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate retry limits and allocate the rolling signature window."""
        if self.max_no_progress < 1:
            raise ValueError("max_no_progress must be positive")
        if self.max_parameter_adaptations < 1:
            raise ValueError("max_parameter_adaptations must be positive")
        self._recent_signatures = deque(maxlen=2 * (self.max_no_progress + 1))

    def record_parameter_adaptation(self, parameter: str) -> str | None:
        """Count bounded L1 changes independently of state signatures.

        Runtime state signatures intentionally exclude call arguments so changing
        an argument cannot keep a failing L1 loop alive indefinitely.

        Args:
            parameter: Name of the argument changed by L1 recovery.

        Returns:
            ``parameter_adaptation_limit`` when the cap was already reached,
            otherwise ``None`` after recording the change.

        Raises:
            ValueError: If ``parameter`` is empty.
        """
        if not parameter.strip():
            raise ValueError("parameter must not be empty")
        count = self.parameter_adaptations.get(parameter, 0)
        if count >= self.max_parameter_adaptations:
            return "parameter_adaptation_limit"
        self.parameter_adaptations[parameter] = count + 1
        return None

    def record_attempt(
        self,
        *,
        state_signature: Hashable | None = None,
        turns: int = 1,
        wall_clock_s: float = 0.0,
        upper_model_calls: int = 0,
        env_steps: int = 1,
    ) -> str | None:
        """Add one attempt and report a reached termination condition.

        Args:
            state_signature: Hashable state identity used for no-progress checks.
            turns: Planner turns consumed by the attempt.
            wall_clock_s: Elapsed wall-clock seconds consumed by the attempt.
            upper_model_calls: Upper-model calls consumed by the attempt.
            env_steps: Environment steps consumed by the attempt.

        Returns:
            The first applicable termination reason, or ``None``.

        Raises:
            ValueError: If any cost increment is negative or has an invalid type.
        """
        if (
            isinstance(turns, bool)
            or not isinstance(turns, int)
            or turns < 0
            or isinstance(upper_model_calls, bool)
            or not isinstance(upper_model_calls, int)
            or upper_model_calls < 0
            or isinstance(env_steps, bool)
            or not isinstance(env_steps, int)
            or env_steps < 0
            or isinstance(wall_clock_s, bool)
            or not isinstance(wall_clock_s, (int, float))
            or not isfinite(wall_clock_s)
            or wall_clock_s < 0
        ):
            raise ValueError("budget increments must be non-negative")
        self.attempts += 1
        self.turns += turns
        self.wall_clock_s += wall_clock_s
        self.upper_model_calls += upper_model_calls
        self.env_steps += env_steps
        self._recent_signatures.append(state_signature)
        self._no_progress_count = (
            self._recent_signatures.count(state_signature) - 1
            if state_signature is not None
            else 0
        )
        reason = self.exhausted_reason()
        if reason is not None:
            self.termination_reason = reason
        return reason

    def exhausted_reason(self) -> str | None:
        """Return the first explicit budget or no-progress termination reason.

        Returns:
            ``no_progress``, ``budget_exhausted``, or ``None``.
        """
        if self._no_progress_count >= self.max_no_progress:
            return "no_progress"
        if self.wall_clock_s >= self.limits.max_wall_clock_s:
            return "budget_exhausted"
        if self.turns >= self.limits.max_turns:
            return "budget_exhausted"
        if self.upper_model_calls >= self.limits.max_upper_model_calls:
            return "budget_exhausted"
        if self.env_steps >= self.limits.max_env_steps:
            return "budget_exhausted"
        return None

    def cost(self) -> dict[str, int | float]:
        """Return the four cost dimensions accepted by recovery decisions.

        Returns:
            Current turns, wall-clock seconds, upper-model calls, and env steps.
        """
        return {
            "turns": self.turns,
            "wall_clock_s": self.wall_clock_s,
            "upper_model_calls": self.upper_model_calls,
            "env_steps": self.env_steps,
        }
