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

"""Separate verification gates for tools, task outcomes, and skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .runtime import RuntimeResult
from .skills import SkillPlaybook
from .tools import ToolExecutor, ToolSpec

Case = tuple[Mapping[str, Any], Mapping[str, Any]]
Check = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Immutable verification outcome, checks, failures, and evidence."""

    subject_id: str
    passed: bool
    checks: Mapping[str, bool] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy mutable container inputs into the frozen report."""
        object.__setattr__(self, "checks", dict(self.checks))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "evidence", dict(self.evidence))


class ToolVerifier:
    """Verify a candidate tool before it is placed in the tool registry."""

    def verify(
        self,
        spec: ToolSpec,
        executor: ToolExecutor,
        cases: Iterable[Case],
        *,
        precondition: Check | None = None,
        postcondition: Check | None = None,
        max_cases: int = 128,
    ) -> VerificationReport:
        """Evaluate a candidate executor over bounded test cases.

        Args:
            spec: Descriptor for the candidate under test.
            executor: Candidate tool implementation.
            cases: Argument and context pairs to execute.
            precondition: Optional gate evaluated before each execution.
            postcondition: Optional gate evaluated on each mapping output.
            max_cases: Maximum number of cases accepted in one report.

        Returns:
            Report containing each evaluated check and failure.

        Raises:
            TypeError: If ``executor`` is not callable.
            ValueError: If ``max_cases`` is less than one.
        """
        if not callable(executor):
            raise TypeError("executor must be callable")
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        checks: dict[str, bool] = {}
        failures: list[str] = []
        count = 0
        for count, (arguments, context) in enumerate(cases, start=1):
            if count > max_cases:
                failures.append(f"case_limit_exceeded:{max_cases}")
                break
            key = f"case_{count}"
            if precondition is not None:
                try:
                    checks[f"{key}.precondition"] = bool(
                        precondition(arguments, context)
                    )
                except Exception as exc:
                    checks[f"{key}.precondition"] = False
                    failures.append(f"{key}.precondition:{type(exc).__name__}")
                if not checks[f"{key}.precondition"]:
                    failures.append(f"{key}.precondition")
                    continue
            try:
                output = executor(arguments, context)
                checks[f"{key}.returns_mapping"] = isinstance(output, Mapping)
            except Exception as exc:
                checks[f"{key}.returns_mapping"] = False
                failures.append(f"{key}.executor:{type(exc).__name__}:{exc}")
                continue
            if not checks[f"{key}.returns_mapping"]:
                failures.append(f"{key}.returns_mapping")
                continue
            if postcondition is not None:
                try:
                    checks[f"{key}.postcondition"] = bool(
                        postcondition(output, context)
                    )
                except Exception as exc:
                    checks[f"{key}.postcondition"] = False
                    failures.append(f"{key}.postcondition:{type(exc).__name__}")
                if not checks[f"{key}.postcondition"]:
                    failures.append(f"{key}.postcondition")
        if count == 0:
            failures.append("no_cases")
        return VerificationReport(
            subject_id=spec.tool_id,
            passed=not failures and bool(checks),
            checks=checks,
            failures=tuple(failures),
            evidence={
                "tool_version": spec.version,
                "case_count": min(count, max_cases),
            },
        )


class SnapshotToolVerifier(ToolVerifier):
    """Verify tools while restoring an injected external-state snapshot."""

    def __init__(self, capture: Callable[[], Any], restore: Callable[[Any], None]):
        """Create a verifier around state capture and restore callables.

        Args:
            capture: Callable invoked once before candidate verification.
            restore: Callable invoked once in a ``finally`` block with the
                captured snapshot.

        Raises:
            TypeError: If either callback is not callable.
        """
        if not callable(capture):
            raise TypeError("capture must be callable")
        if not callable(restore):
            raise TypeError("restore must be callable")
        self._capture = capture
        self._restore = restore

    def verify(
        self,
        spec: ToolSpec,
        executor: ToolExecutor,
        cases: Iterable[Case],
        *,
        precondition: Check | None = None,
        postcondition: Check | None = None,
        max_cases: int = 128,
    ) -> VerificationReport:
        """Verify a tool and restore the captured state on every exit path."""
        snapshot = self._capture()
        try:
            return super().verify(
                spec,
                executor,
                cases,
                precondition=precondition,
                postcondition=postcondition,
                max_cases=max_cases,
            )
        finally:
            self._restore(snapshot)


class OutcomeVerifier:
    """Verify the task-level outcome after a skill runtime execution."""

    def verify(
        self,
        result: RuntimeResult,
        checks: Mapping[str, Callable[[RuntimeResult], bool]],
    ) -> VerificationReport:
        """Evaluate runtime success and named task-level checks.

        Args:
            result: Completed skill runtime result.
            checks: Named predicates evaluated against ``result``.

        Returns:
            Report containing runtime success and each task-level check.

        Raises:
            ValueError: If a check name is empty.
        """
        evaluated: dict[str, bool] = {"runtime_success": result.success}
        failures: list[str] = [] if result.success else ["runtime_success"]
        for name, check in checks.items():
            if not name.strip():
                raise ValueError("outcome check names must not be empty")
            try:
                evaluated[name] = bool(check(result))
            except Exception as exc:
                evaluated[name] = False
                failures.append(f"{name}:{type(exc).__name__}")
            if not evaluated[name] and name not in failures:
                failures.append(name)
        return VerificationReport(
            subject_id=result.skill_id,
            passed=not failures and all(evaluated.values()),
            checks=evaluated,
            failures=tuple(failures),
            evidence={"tool_call_count": len(result.tool_results)},
        )


class SkillVerifier:
    """Verify an experiential playbook over replay/transfer cases."""

    def verify(
        self,
        skill: SkillPlaybook,
        evaluations: Mapping[str, Callable[[SkillPlaybook], bool]],
    ) -> VerificationReport:
        """Evaluate all required replay and transfer gates.

        Args:
            skill: Playbook being validated.
            evaluations: Predicates for replay, perturbation, held-out, and
                regression checks.

        Returns:
            Report containing all four required checks.

        Raises:
            ValueError: If a required evaluation is missing.
        """
        required = ("replay", "perturbation", "held_out", "regression")
        missing = [name for name in required if name not in evaluations]
        if missing:
            raise ValueError(f"missing skill verification checks: {', '.join(missing)}")
        checks: dict[str, bool] = {}
        failures: list[str] = []
        for name in required:
            try:
                checks[name] = bool(evaluations[name](skill))
            except Exception as exc:
                checks[name] = False
                failures.append(f"{name}:{type(exc).__name__}")
            if not checks[name] and name not in failures:
                failures.append(name)
        return VerificationReport(skill.skill_id, not failures, checks, tuple(failures))
