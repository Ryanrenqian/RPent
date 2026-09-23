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

"""A small runtime that proves skills orchestrate tools without owning source."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Mapping

from .adapt import ParameterAdapter
from .budget import BudgetLedger
from .diagnose import DiagnosisSignals, FailureDiagnoser
from .events import (
    ExecutionEvent,
    FailureEvent,
    RecoveryAction,
    RecoveryDecision,
    RecoveryLevel,
)
from .libero_evidence import map_libero_evidence
from .router import FailureRouter
from .skills import SkillPlaybook
from .tools import ToolCall, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from .synthesis import ToolSynthesizer
    from .tool_gap import ToolGapAdapter
    from .verification import ToolVerifier


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Final skill outcome with tool results, events, decision, and costs."""

    success: bool
    skill_id: str
    tool_results: tuple[ToolResult, ...] = ()
    events: tuple[ExecutionEvent, ...] = ()
    decision: RecoveryDecision | None = None
    error: str | None = None
    ledger: BudgetLedger | None = None


class SkillRuntime:
    """Execute a playbook by resolving tool IDs from a registry.

    L1 adaptation changes the next call's arguments.  Runtime state signatures
    deliberately exclude arguments, and ``BudgetLedger`` separately caps each
    parameter's adaptations, so argument changes cannot defeat no-progress
    termination.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        router: FailureRouter | None = None,
        diagnoser: FailureDiagnoser | None = None,
        ledger: BudgetLedger | None = None,
        recovery_loop: bool = False,
        parameter_adapter: ParameterAdapter | None = None,
        tool_gap_adapter: ToolGapAdapter | None = None,
        synthesizer: ToolSynthesizer | None = None,
        tool_verifier: ToolVerifier | None = None,
    ) -> None:
        """Configure tool resolution and optional recovery collaborators.

        Args:
            tools: Registry-like object used for lookup and invocation.
            router: Optional failure router override.
            diagnoser: Optional failure diagnoser override.
            ledger: Optional reusable budget ledger.
            recovery_loop: Whether failures should loop through recovery.
            parameter_adapter: Optional L1 parameter adapter override.
            tool_gap_adapter: Optional L3 handoff and synthesis adapter.
            synthesizer: Optional L3 candidate synthesis backend.
            tool_verifier: Explicit verifier responsible for candidate execution.
        """
        self.tools = tools
        self.router = router or FailureRouter()
        self.diagnoser = diagnoser or FailureDiagnoser()
        self.ledger = ledger
        self.recovery_loop = recovery_loop
        self.parameter_adapter = parameter_adapter or ParameterAdapter()
        self.tool_gap_adapter = tool_gap_adapter
        self.synthesizer = synthesizer
        self.tool_verifier = tool_verifier

    def execute(
        self,
        skill: SkillPlaybook,
        *,
        episode_id: str,
        state: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        recovery_loop: bool | None = None,
    ) -> RuntimeResult:
        """Execute every playbook step with bounded recovery.

        Args:
            skill: Verified playbook to execute.
            episode_id: Identifier attached to emitted events.
            state: Initial runtime state and fallback diagnosis evidence.
            parameters: Arguments merged into every step binding.
            recovery_loop: Per-call override for failure recovery looping.

        Returns:
            Terminal runtime result with accumulated events, calls, and costs.
        """
        current_state = dict(state or {})
        params = dict(parameters or {})
        results: list[ToolResult] = []
        events: list[ExecutionEvent] = []
        ledger = self.ledger or BudgetLedger()
        use_recovery_loop = (
            self.recovery_loop if recovery_loop is None else recovery_loop
        )
        last_adaptation_evidence: dict[str, Any] = {}

        def contextualize(
            decision: RecoveryDecision,
            event: FailureEvent,
            termination_reason: str | None = None,
        ) -> RecoveryDecision:
            return RecoveryDecision(
                decision.level,
                decision.action,
                decision.reason,
                decision.evidence,
                source=decision.source,
                event_id=event.event_id,
                episode_id=event.episode_id,
                step=event.step,
                cost=ledger.cost(),
                termination_reason=termination_reason,
                attempts=ledger.attempts,
            )

        def give_up(
            event: FailureEvent,
            reason: str,
            evidence_extra: Mapping[str, Any] | None = None,
        ) -> RuntimeResult:
            terminal = FailureEvent(
                episode_id=event.episode_id,
                goal=event.goal,
                step=event.step,
                state=event.state,
                completed_subgoals=event.completed_subgoals,
                tool_id=event.tool_id,
                outcome="give_up",
                failure_family=event.failure_family,
                diagnosis=event.diagnosis,
                retryable=event.retryable,
                parameter_issue=event.parameter_issue,
                world_state_invalidated=event.world_state_invalidated,
                required_capability=event.required_capability,
                termination_reason=reason,
            )
            events.append(terminal)
            decision = RecoveryDecision(
                RecoveryLevel.UNKNOWN,
                RecoveryAction.GIVE_UP,
                f"recovery terminated: {reason}",
                {
                    "failure_family": event.failure_family,
                    "evidence_sufficient": event.diagnosis.get(
                        "evidence_sufficiency", {}
                    ).get("value", False),
                    **last_adaptation_evidence,
                    **dict(evidence_extra or {}),
                },
                event_id=terminal.event_id,
                episode_id=terminal.episode_id,
                step=terminal.step,
                cost=ledger.cost(),
                termination_reason=reason,
                attempts=ledger.attempts,
            )
            return RuntimeResult(
                False,
                skill.skill_id,
                tuple(results),
                tuple(events),
                decision,
                event.diagnosis.get("tool_error", {}).get("value")
                if isinstance(event.diagnosis.get("tool_error"), Mapping)
                else None,
                ledger,
            )

        def synthesize_l3(
            event: FailureEvent,
            arguments: Mapping[str, Any],
            prior_reason: str | None,
        ) -> RuntimeResult | str:
            if prior_reason is not None:
                return give_up(event, prior_reason)
            if self.tool_verifier is None:
                synthesis_reason = ledger.record_attempt(
                    state_signature=(event.tool_id, "synthesis_without_sandbox"),
                    turns=0,
                    env_steps=0,
                )
                return give_up(
                    event,
                    synthesis_reason
                    or "L3 synthesis requires an explicit verification sandbox",
                    {"synthesis_status": "verification_refused_sandbox_missing"},
                )

            synthesis_started = monotonic()
            try:
                synthesis = self.tool_gap_adapter.synthesize_tool_gap(
                    event,
                    self.synthesizer,
                    [(dict(arguments), dict(current_state))],
                    verifier=self.tool_verifier,
                )
            except Exception as exc:
                synthesis_reason = ledger.record_attempt(
                    state_signature=(event.tool_id, "synthesis_failed"),
                    wall_clock_s=monotonic() - synthesis_started,
                    upper_model_calls=1,
                    env_steps=0,
                )
                return give_up(
                    event,
                    synthesis_reason or f"tool synthesis failed: {type(exc).__name__}",
                    {
                        "synthesis_status": "failed",
                        "synthesis_error": str(exc),
                    },
                )

            synthesis_reason = ledger.record_attempt(
                state_signature=(event.tool_id, "synthesis"),
                wall_clock_s=monotonic() - synthesis_started,
                upper_model_calls=1,
                env_steps=0,
            )
            synthesis_evidence = {
                "synthesis_status": "registered"
                if synthesis.registered
                else "rejected",
                "candidate_tool_id": synthesis.candidate.spec.tool_id,
                "verification_passed": synthesis.verification.passed,
                "verification_failures": list(synthesis.verification.failures),
                "registered": synthesis.registered,
            }
            if synthesis_reason is not None:
                return give_up(event, synthesis_reason, synthesis_evidence)
            if not synthesis.registered:
                return give_up(
                    event,
                    "tool synthesis verification or registration failed",
                    synthesis_evidence,
                )
            return synthesis.candidate.spec.tool_id

        for index, step in enumerate(skill.steps):
            attempt_arguments = dict(step.argument_bindings)
            attempt_arguments.update(params)
            active_tool_id = step.tool_id
            while True:
                call = ToolCall(tool_id=active_tool_id, arguments=attempt_arguments)
                started = ExecutionEvent(
                    episode_id=episode_id,
                    goal=skill.goal,
                    step=index,
                    state=current_state,
                    completed_subgoals=tuple(s.step_id for s in skill.steps[:index]),
                    tool_id=active_tool_id,
                    outcome="started",
                )
                events.append(started)
                if not self.tools.has(active_tool_id):
                    signals = DiagnosisSignals(
                        scoreable=True,
                        cell_input={"scoreable": True},
                        libero_predicate=None,
                        missing_capability=active_tool_id,
                    )
                    result = self.diagnoser.diagnose(signals)
                    gap = result.to_failure_event(
                        started,
                        outcome="tool_gap",
                        tool_gap=True,
                        missing_capability=active_tool_id,
                    )
                    events.append(gap)
                    reason = ledger.record_attempt(
                        state_signature=(active_tool_id, "missing"), env_steps=1
                    )
                    decision = self.router.route(
                        gap,
                        available_tool_ids=(
                            spec.tool_id for spec in self.tools.list_specs()
                        ),
                    )
                    if (
                        use_recovery_loop
                        and decision.level is RecoveryLevel.L3
                        and self.tool_gap_adapter is not None
                        and self.synthesizer is not None
                    ):
                        synthesis_outcome = synthesize_l3(
                            gap, attempt_arguments, reason
                        )
                        if isinstance(synthesis_outcome, RuntimeResult):
                            return synthesis_outcome
                        active_tool_id = synthesis_outcome
                        continue
                    return RuntimeResult(
                        False,
                        skill.skill_id,
                        tuple(results),
                        tuple(events),
                        contextualize(decision, gap),
                        f"missing tool: {active_tool_id}",
                        ledger,
                    )
                invocation_start = monotonic()
                result = self.tools.invoke(call, current_state)
                measured_wall_clock_s = monotonic() - invocation_start
                results.append(result)
                metadata = result.metadata
                wall_clock_s = max(
                    measured_wall_clock_s, float(metadata.get("wall_clock_s", 0.0))
                )
                if not result.success:
                    output = result.output if isinstance(result.output, Mapping) else {}
                    mapped_evidence = map_libero_evidence(output)
                    metadata = result.metadata
                    signals = DiagnosisSignals(
                        scoreable=True,
                        cell_input={"scoreable": True},
                        libero_predicate=None,
                        tool_error=result.error,
                        end_effector_pose=metadata.get(
                            "end_effector_pose",
                            mapped_evidence.get(
                                "end_effector_pose",
                                output.get(
                                    "end_effector_pose",
                                    output.get(
                                        "pose", current_state.get("end_effector_pose")
                                    ),
                                ),
                            ),
                        ),
                        gripper_opening=metadata.get(
                            "gripper_opening",
                            mapped_evidence.get(
                                "gripper_opening", current_state.get("gripper_opening")
                            ),
                        ),
                        transcript_text=metadata.get(
                            "transcript_text",
                            output.get(
                                "transcript_text", current_state.get("transcript_text")
                            ),
                        ),
                    )
                    diagnosis = self.diagnoser.diagnose(signals)
                    failure = diagnosis.to_failure_event(
                        ExecutionEvent(
                            episode_id=episode_id,
                            goal=skill.goal,
                            step=index,
                            state=current_state,
                            completed_subgoals=tuple(
                                s.step_id for s in skill.steps[:index]
                            ),
                            tool_id=active_tool_id,
                            outcome="failed",
                        )
                    )
                    events.append(failure)
                    reason = ledger.record_attempt(
                        # Tool error text can echo the adapted arguments.  Keep
                        # it out of the progress signature so L1 cannot evade
                        # no-progress detection by changing its own message.
                        state_signature=(
                            active_tool_id,
                            repr(
                                sorted(
                                    current_state.items(),
                                    key=lambda item: repr(item[0]),
                                )
                            ),
                        ),
                        turns=int(metadata.get("turns", 1)),
                        wall_clock_s=wall_clock_s,
                        upper_model_calls=int(metadata.get("upper_model_calls", 0)),
                        env_steps=int(metadata.get("env_steps", 1)),
                    )
                    decision = self.router.route(
                        failure,
                        available_tool_ids=(s.tool_id for s in self.tools.list_specs()),
                    )
                    if (
                        use_recovery_loop
                        and decision.level is RecoveryLevel.L3
                        and self.tool_gap_adapter is not None
                        and self.synthesizer is not None
                    ):
                        synthesis_outcome = synthesize_l3(
                            failure, attempt_arguments, reason
                        )
                        if isinstance(synthesis_outcome, RuntimeResult):
                            return synthesis_outcome
                        active_tool_id = synthesis_outcome
                        continue
                    if use_recovery_loop and decision.level == RecoveryLevel.L1:
                        if reason is not None:
                            return give_up(failure, reason)
                        adaptation = self.parameter_adapter.adapt(
                            diagnosis, attempt_arguments
                        )
                        if not adaptation.adaptation_available or not adaptation.delta:
                            return give_up(
                                failure,
                                "L1 无可用自适应证据",
                                {
                                    "adaptation_available": False,
                                    "adaptation_reason": adaptation.reason,
                                },
                            )
                        for parameter in adaptation.delta:
                            adaptation_reason = ledger.record_parameter_adaptation(
                                parameter
                            )
                            if adaptation_reason is not None:
                                return give_up(
                                    failure,
                                    adaptation_reason,
                                    {
                                        "adaptation_available": True,
                                        "adaptation_delta": dict(adaptation.delta),
                                        "adaptation_evidence": dict(
                                            adaptation.evidence
                                        ),
                                    },
                                )
                        old_arguments = dict(attempt_arguments)
                        attempt_arguments.update(adaptation.delta)
                        if attempt_arguments == old_arguments:
                            return give_up(
                                failure,
                                "L1 adaptation produced no argument change",
                                {
                                    "adaptation_available": False,
                                    "adaptation_delta": dict(adaptation.delta),
                                },
                            )
                        last_adaptation_evidence.clear()
                        last_adaptation_evidence.update(
                            {
                                "adaptation_available": True,
                                "adaptation_delta": dict(adaptation.delta),
                                "adaptation_evidence": dict(adaptation.evidence),
                                "adaptation_reason": adaptation.reason,
                            }
                        )
                        continue
                    if use_recovery_loop and decision.level in {
                        RecoveryLevel.L0,
                        RecoveryLevel.L2,
                    }:
                        if reason is not None:
                            return give_up(failure, reason)
                        continue
                    return RuntimeResult(
                        False,
                        skill.skill_id,
                        tuple(results),
                        tuple(events),
                        contextualize(decision, failure),
                        result.error,
                        ledger,
                    )
                current_state.update(result.output)
                step_reason = ledger.record_attempt(
                    state_signature=(
                        active_tool_id,
                        repr(
                            sorted(
                                current_state.items(), key=lambda item: repr(item[0])
                            )
                        ),
                    ),
                    turns=int(metadata.get("turns", 1)),
                    wall_clock_s=wall_clock_s,
                    upper_model_calls=int(metadata.get("upper_model_calls", 0)),
                    env_steps=int(metadata.get("env_steps", 1)),
                )
                events.append(
                    ExecutionEvent(
                        episode_id=episode_id,
                        goal=skill.goal,
                        step=index,
                        state=current_state,
                        completed_subgoals=tuple(
                            s.step_id for s in skill.steps[: index + 1]
                        ),
                        tool_id=active_tool_id,
                        outcome="succeeded",
                    )
                )
                if step_reason is not None and index < len(skill.steps) - 1:
                    budget_diagnosis = self.diagnoser.diagnose(
                        DiagnosisSignals(
                            scoreable=True,
                            cell_input={
                                "scoreable": True,
                                "scored_reason": step_reason,
                            },
                            libero_predicate=None,
                            transcript_text=step_reason,
                        )
                    )
                    budget_failure = budget_diagnosis.to_failure_event(
                        ExecutionEvent(
                            episode_id=episode_id,
                            goal=skill.goal,
                            step=index,
                            state=current_state,
                            completed_subgoals=tuple(
                                s.step_id for s in skill.steps[: index + 1]
                            ),
                            tool_id=active_tool_id,
                            outcome="failed",
                        ),
                        termination_reason=step_reason,
                    )
                    events.append(budget_failure)
                    return give_up(budget_failure, step_reason)
                break
        return RuntimeResult(
            True, skill.skill_id, tuple(results), tuple(events), ledger=ledger
        )
