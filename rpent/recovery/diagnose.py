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

"""Failure diagnosis from explicit runtime signals.

The audit fixture shipped with this module is synthetic, not human annotated;
it exists only for structural and regression checks.  The human-labelled audit
set requested by AGENTS.md P-2 has not yet been obtained and must not support a
scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .events import ExecutionEvent, FailureEvent, ToolGapEvent


def _evidence(source: str, value: Any) -> dict[str, Any]:
    return {"source": source, "value": value}


@dataclass(frozen=True, slots=True)
class DiagnosisSignals:
    """Signals available to the diagnoser; absent sensors remain explicitly None."""

    cell_input: Mapping[str, Any] = field(default_factory=dict)
    libero_predicate: bool | None = None
    pi05_heuristic: Any = None
    sam3_observation: Any = None
    end_effector_pose: Any = None
    gripper_opening: Any = None
    transcript_text: str | None = None
    tool_error: str | None = None
    missing_capability: str | None = None
    libero_tool_evidence: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scored_reason_source: str | None = None
    _cell_record_evidence: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Enforce mapping-shaped cell input."""
        if not isinstance(self.cell_input, Mapping):
            raise TypeError("cell_input must be a mapping")

    @classmethod
    def from_cell_input(
        cls, cell_input: Mapping[str, Any], **signals: Any
    ) -> "DiagnosisSignals":
        """Build signals from explicitly scoreable cell input.

        Args:
            cell_input: Mapping produced for a scoreable cell.
            **signals: Runtime signals that override or supplement cell data.

        Returns:
            Validated diagnosis signals.

        Raises:
            ValueError: If the input is not explicitly marked scoreable.
        """
        scoreable = signals.pop("scoreable", cell_input.get("scoreable"))
        if scoreable is not True:
            raise ValueError("cell_input must be explicitly marked scoreable=True")
        predicate = signals.pop("libero_predicate", cell_input.get("predicate"))
        return cls(
            cell_input=dict(cell_input),
            libero_predicate=predicate,
            **signals,
        )

    @classmethod
    def from_cell_record_input(
        cls, cell_input: Mapping[str, Any], **signals: Any
    ) -> "DiagnosisSignals":
        """Consume the non-None output of CellRecord.to_failure_event_input().

        That producer already filters unscorable and successful cells; this
        named adapter makes that provenance explicit while retaining the
        stricter ``from_cell_input`` gate for arbitrary mappings.

        Args:
            cell_input: Non-``None`` failure input produced by a cell record.
            **signals: Runtime signals that supplement the cell input.

        Returns:
            Validated diagnosis signals.

        Raises:
            ValueError: If the producer supplied no mapping.
        """
        if cell_input is None or not isinstance(cell_input, Mapping):
            raise ValueError(
                "CellRecord.to_failure_event_input() returned no scoreable mapping"
            )
        instance = cls.from_cell_input(cell_input, scoreable=True, **signals)
        object.__setattr__(instance, "_cell_record_evidence", True)
        return instance


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    """Auditable diagnosis and the four router fields it is authoritative for."""

    failure_family: str
    evidence: Mapping[str, Mapping[str, Any]]
    retryable: bool
    parameter_issue: bool
    world_state_invalidated: bool
    required_capability: str | None
    sufficient: bool
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the failure family and evidence envelope shape."""
        if not self.failure_family.strip():
            raise ValueError("failure_family must not be empty")
        for item in self.evidence.values():
            if set(item) != {"source", "value"}:
                raise ValueError("diagnosis evidence must contain source and value")

    def to_failure_event(
        self,
        base: ExecutionEvent,
        *,
        outcome: str = "failed",
        tool_gap: bool = False,
        termination_reason: str | None = None,
        missing_capability: str | None = None,
    ) -> FailureEvent:
        """Apply this diagnosis to an event without re-diagnosing it.

        Args:
            base: Execution event whose context is retained.
            outcome: Outcome assigned to the failure event.
            tool_gap: Whether to create a specialized tool-gap event.
            termination_reason: Optional override for the diagnosis termination.
            missing_capability: Capability recorded on a tool-gap event.

        Returns:
            A failure event, specialized as a tool gap when requested.
        """
        kwargs = {
            "episode_id": base.episode_id,
            "goal": base.goal,
            "step": base.step,
            "state": base.state,
            "completed_subgoals": base.completed_subgoals,
            "tool_id": base.tool_id,
            "outcome": outcome,
            "timestamp": base.timestamp,
            "metadata": base.metadata,
            "failure_family": self.failure_family,
            "diagnosis": dict(self.evidence),
            "retryable": self.retryable,
            "parameter_issue": self.parameter_issue,
            "world_state_invalidated": self.world_state_invalidated,
            "required_capability": self.required_capability,
            "termination_reason": termination_reason or self.termination_reason,
        }
        if tool_gap:
            return ToolGapEvent(
                **kwargs,
                missing_capability=missing_capability
                or self.required_capability
                or "unknown",
            )
        return FailureEvent(**kwargs)


class FailureDiagnoser:
    """Convert signals into evidence by fixed priority.

    Priority is registry gap, SAM3, LIBERO tool-result evidence, pose/gripper,
    transcript, tool error, scored end reason, then Pi0.5 self-report.
    Self-report alone never sets a recovery-routing boolean.
    """

    def diagnose(
        self, signals: DiagnosisSignals | Mapping[str, Any], **signal_overrides: Any
    ) -> DiagnosisResult:
        """Diagnose failure signals and retain source-bearing evidence.

        Args:
            signals: Validated online signals or a scoreable cell mapping.
            **signal_overrides: Runtime values accepted only with a cell mapping.

        Returns:
            Failure family, routing flags, sufficiency, and all evidence channels.

        Raises:
            TypeError: If overrides accompany an existing signal object.
            ValueError: If a cell mapping is unscorable or has an unknown scored reason.
        """
        if isinstance(signals, Mapping):
            signals = DiagnosisSignals.from_cell_input(signals, **signal_overrides)
        elif signal_overrides:
            raise TypeError("signal overrides are only valid with cell_input mappings")
        evidence = {
            "libero_predicate": _evidence(
                "LIBERO states.json.steps[*].terminated", signals.libero_predicate
            ),
            "pi05_heuristic": _evidence(
                "Pi0.5 heuristic self-report", signals.pi05_heuristic
            ),
            "sam3_observation": _evidence("SAM3 observation", signals.sam3_observation),
            "end_effector_pose": _evidence(
                "end-effector pose", signals.end_effector_pose
            ),
            "gripper_opening": _evidence("gripper opening", signals.gripper_opening),
            "transcript_text": _evidence("transcript text", signals.transcript_text),
            "tool_error": _evidence("tool result error", signals.tool_error),
            "missing_capability": _evidence(
                "tool registry lookup", signals.missing_capability
            ),
            "scored_reason": _evidence(
                (
                    "CellRecord.scored_reason"
                    if signals._cell_record_evidence
                    else signals.scored_reason_source or "diagnosis cell input"
                ),
                signals.cell_input.get("scored_reason"),
            ),
        }
        for name, item in signals.libero_tool_evidence.items():
            if name in evidence:
                raise ValueError(f"LIBERO tool evidence conflicts with {name!r}")
            if set(item) != {"source", "value"}:
                raise ValueError("LIBERO tool evidence must contain source and value")
            evidence[name] = dict(item)
        if signals._cell_record_evidence:
            evidence["scoreable"] = _evidence("CellRecord.scoreable", True)
        scored_reason = signals.cell_input.get("scored_reason")
        if scored_reason not in {
            None,
            "budget_exhausted",
            "agent_gave_up",
            "no_progress",
        }:
            raise ValueError(f"unexpected scored_reason: {scored_reason!r}")
        heuristic = (
            signals.pi05_heuristic
            if isinstance(signals.pi05_heuristic, Mapping)
            else {}
        )
        text = (signals.transcript_text or "").lower()
        sam3 = (
            signals.sam3_observation
            if isinstance(signals.sam3_observation, Mapping)
            else {}
        )
        pose = (
            signals.end_effector_pose
            if isinstance(signals.end_effector_pose, Mapping)
            else {}
        )
        libero_values = {
            name: item["value"]
            for name, item in signals.libero_tool_evidence.items()
            if isinstance(item.get("value"), Mapping)
        }
        family = None
        parameter = False
        world = False
        retryable = False
        required = signals.missing_capability
        rule_id = None

        if required:
            family, rule_id = "missing_capability", "registry_missing_capability"
        elif sam3.get("blocked") or sam3.get("object_displaced"):
            family, world, rule_id = "world_state_invalidated", True, "sam3_world_state"
        elif libero_values.get("libero_position", {}).get("reached") is False:
            family, parameter, rule_id = (
                "parameter_or_pose",
                True,
                "libero_position_not_reached",
            )
        elif libero_values.get("libero_orientation", {}).get("reached") is False:
            family, parameter, rule_id = (
                "parameter_or_pose",
                True,
                "libero_orientation_not_reached",
            )
        elif libero_values.get("libero_pick_descent", {}).get("reached") is False:
            family, parameter, rule_id = (
                "parameter_or_pose",
                True,
                "libero_pick_no_descent",
            )
        elif libero_values.get("libero_pick_close", {}).get("reached") is False:
            family, parameter, rule_id = (
                "grasp_contact",
                True,
                "libero_pick_no_close",
            )
        elif libero_values.get("libero_pick_lift", {}).get("reached") is False:
            family, parameter, rule_id = (
                "grasp_contact",
                True,
                "libero_pick_no_lift",
            )
        elif (
            pose.get("reachable") is False
            or "wrong pose" in text
            or "misaligned" in text
        ):
            family, parameter, rule_id = (
                "parameter_or_pose",
                True,
                "pose_or_transcript_alignment",
            )
        else:
            opening = signals.gripper_opening
            if isinstance(opening, Mapping):
                opening = opening.get("value", opening.get("opening"))
            gripper_failure = (
                isinstance(opening, (int, float))
                and not isinstance(opening, bool)
                and opening <= 0
                and ("grasp" in text or "pick" in text)
            )
        if family is None and gripper_failure:
            family, parameter, rule_id = (
                "grasp_contact",
                True,
                "gripper_and_transcript_grasp",
            )
        elif family is None and any(
            token in text
            for token in ("blocked", "occluded", "displaced", "state changed")
        ):
            family, world, rule_id = (
                "world_state_invalidated",
                True,
                "transcript_world_state",
            )
        elif family is None and signals.tool_error:
            family, retryable, rule_id = "tool_execution", True, "tool_error"
        elif family is None and scored_reason:
            family, rule_id = scored_reason, "scored_end_reason"
        elif family is None and heuristic.get("required_capability"):
            required = str(heuristic["required_capability"])
            family, rule_id = "missing_capability", "heuristic_required_capability"
            evidence["required_capability"] = _evidence(
                "Pi0.5 heuristic self-report.required_capability", required
            )
        elif family is None and heuristic.get("failure_family"):
            family, rule_id = (
                str(heuristic["failure_family"]),
                "heuristic_failure_family",
            )

        sufficient = family is not None
        if not sufficient:
            family = "unknown"
            parameter = world = retryable = False
        evidence["rule_id"] = _evidence("FailureDiagnoser decisive rule", rule_id)
        evidence["evidence_sufficiency"] = _evidence(
            "FailureDiagnoser evidence sufficiency", sufficient
        )
        termination_reason = scored_reason
        return DiagnosisResult(
            family,
            evidence,
            retryable,
            parameter,
            world,
            required,
            sufficient,
            termination_reason,
        )
