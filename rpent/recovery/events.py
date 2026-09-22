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

"""Machine-readable events and routing decisions for the evolution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from time import time
from typing import Any, Mapping
from uuid import uuid4


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


class RecoveryLevel(str, Enum):
    """Auditable failure severity used by the first rule-based router."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(str, Enum):
    """Actions the recovery router can select."""

    RETRY = "retry"
    ADAPT = "adapt_parameters"
    REPLAN = "replan"
    SYNTHESIZE_TOOL = "synthesize_tool"
    GIVE_UP = "give_up"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """A normalized observation of one execution transition."""

    episode_id: str
    goal: str
    step: int
    state: Mapping[str, Any] = field(default_factory=dict)
    completed_subgoals: tuple[str, ...] = ()
    tool_id: str | None = None
    outcome: str = "observed"
    timestamp: float = field(default_factory=time)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        """Validate identifiers, step, timestamp, and copy container inputs."""
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.outcome.strip():
            raise ValueError("outcome must not be empty")
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if not isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "state", _copy_mapping(self.state))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        object.__setattr__(self, "completed_subgoals", tuple(self.completed_subgoals))

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this event to JSON-compatible values.

        Returns:
            Manifest containing the complete execution event.
        """
        return {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "goal": self.goal,
            "step": self.step,
            "state": _json_value(self.state),
            "completed_subgoals": list(self.completed_subgoals),
            "tool_id": self.tool_id,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ExecutionEvent:
        """Restore an execution event from a manifest.

        Args:
            payload: Serialized execution event fields.

        Returns:
            Validated execution event.
        """
        return cls(
            episode_id=payload["episode_id"],
            goal=payload["goal"],
            step=payload["step"],
            state=payload.get("state", {}),
            completed_subgoals=tuple(payload.get("completed_subgoals", ())),
            tool_id=payload.get("tool_id"),
            outcome=payload.get("outcome", "observed"),
            timestamp=payload["timestamp"],
            metadata=payload.get("metadata", {}),
            event_id=payload["event_id"],
        )


@dataclass(frozen=True, slots=True)
class FailureEvent(ExecutionEvent):
    """A failed execution with explicit evidence for routing."""

    failure_family: str = "unknown"
    diagnosis: Mapping[str, Any] = field(default_factory=dict)
    retryable: bool = False
    parameter_issue: bool = False
    world_state_invalidated: bool = False
    required_capability: str | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate failure fields after validating the base event."""
        ExecutionEvent.__post_init__(self)
        if not self.failure_family.strip():
            raise ValueError("failure_family must not be empty")
        if self.termination_reason is not None and not self.termination_reason.strip():
            raise ValueError("termination_reason must not be empty")
        object.__setattr__(self, "diagnosis", _copy_mapping(self.diagnosis))

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this failure and its base execution fields.

        Returns:
            JSON-compatible failure event manifest.
        """
        return {
            **ExecutionEvent.to_manifest(self),
            "failure_family": self.failure_family,
            "diagnosis": _json_value(self.diagnosis),
            "retryable": self.retryable,
            "parameter_issue": self.parameter_issue,
            "world_state_invalidated": self.world_state_invalidated,
            "required_capability": self.required_capability,
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> FailureEvent:
        """Restore a failure event from a manifest.

        Args:
            payload: Serialized failure and execution fields.

        Returns:
            Validated failure event.
        """
        return cls(
            **ExecutionEvent.from_manifest(payload).to_manifest(),
            failure_family=payload.get("failure_family", "unknown"),
            diagnosis=payload.get("diagnosis", {}),
            retryable=payload.get("retryable", False),
            parameter_issue=payload.get("parameter_issue", False),
            world_state_invalidated=payload.get("world_state_invalidated", False),
            required_capability=payload.get("required_capability"),
            termination_reason=payload.get("termination_reason"),
        )


@dataclass(frozen=True, slots=True)
class ToolGapEvent(FailureEvent):
    """A failure specifically indicating that a callable tool is missing."""

    missing_capability: str = ""

    def __post_init__(self) -> None:
        """Validate the missing capability after base failure validation."""
        FailureEvent.__post_init__(self)
        if not self.missing_capability.strip():
            raise ValueError("missing_capability must not be empty")

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this tool-gap event.

        Returns:
            JSON-compatible tool-gap manifest.
        """
        return {
            **FailureEvent.to_manifest(self),
            "missing_capability": self.missing_capability,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ToolGapEvent:
        """Restore a tool-gap event from a manifest.

        Args:
            payload: Serialized tool-gap and failure fields.

        Returns:
            Validated tool-gap event.
        """
        return cls(
            **FailureEvent.from_manifest(payload).to_manifest(),
            missing_capability=payload["missing_capability"],
        )


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """The router's auditable decision and the evidence behind it."""

    level: RecoveryLevel
    action: RecoveryAction
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    source: str = "rule"
    event_id: str | None = None
    episode_id: str | None = None
    step: int | None = None
    cost: Mapping[str, Any] = field(default_factory=dict)
    termination_reason: str | None = None
    attempts: int = 0

    def __post_init__(self) -> None:
        """Validate decision provenance, cost, and attempt metadata."""
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.source not in {"rule", "oracle"}:
            raise ValueError("source must be 'rule' or 'oracle'")
        for name in ("event_id", "episode_id", "termination_reason"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.step is not None and self.step < 0:
            raise ValueError("step must be non-negative")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        allowed = {"turns", "wall_clock_s", "upper_model_calls", "env_steps"}
        if set(self.cost) - allowed:
            raise ValueError("cost contains unknown keys")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            for value in self.cost.values()
        ):
            raise ValueError("cost values must be non-negative numbers")
        object.__setattr__(self, "evidence", _copy_mapping(self.evidence))
        object.__setattr__(self, "cost", _copy_mapping(self.cost))

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this routing decision.

        Returns:
            JSON-compatible recovery decision manifest.
        """
        return {
            "level": self.level.value,
            "action": self.action.value,
            "reason": self.reason,
            "evidence": _json_value(self.evidence),
            "source": self.source,
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "cost": _json_value(self.cost),
            "termination_reason": self.termination_reason,
            "attempts": self.attempts,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> RecoveryDecision:
        """Restore a recovery decision from a manifest.

        Args:
            payload: Serialized decision fields.

        Returns:
            Validated recovery decision.
        """
        return cls(
            level=RecoveryLevel(payload["level"]),
            action=RecoveryAction(payload["action"]),
            reason=payload["reason"],
            evidence=payload.get("evidence", {}),
            source=payload.get("source", "rule"),
            event_id=payload.get("event_id"),
            episode_id=payload.get("episode_id"),
            step=payload.get("step"),
            cost=payload.get("cost", {}),
            termination_reason=payload.get("termination_reason"),
            attempts=payload.get("attempts", 0),
        )
