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

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rpent.recovery import (
    DiagnosisSignals,
    FailureDiagnoser,
    FailureEvent,
    FailureRouter,
    RecoveryLevel,
    SkillPlaybook,
    SkillStep,
    SkillValidation,
    diagnoser_confusion_matrix,
)


def validated(skill_id: str = "recover-grasp") -> SkillPlaybook:
    return SkillPlaybook(
        skill_id=skill_id,
        name="Recover grasp",
        goal="pick object",
        trigger_labels=frozenset({"failed_grasp", "object_displaced"}),
        diagnosis="grasp failed but object remains reachable",
        steps=(SkillStep("restage", "move object to reachable pose", "restage-tool"),),
        verification_checks=("object grasped",),
        provenance={"episode_id": "ep-1"},
        validation=SkillValidation("e-1", True, True, True, True),
    )


class TestDiagnoser:
    def test_scoreable_gate_and_source_bearing_evidence(self):
        diagnoser = FailureDiagnoser()
        with pytest.raises(ValueError):
            diagnoser.diagnose({"predicate": False}, scoreable=False)
        result = diagnoser.diagnose(
            {"predicate": False, "scored_reason": "budget_exhausted"},
            scoreable=True,
            sam3_observation={"blocked": True},
        )
        assert result.failure_family == "world_state_invalidated"
        assert result.termination_reason == "budget_exhausted"
        assert all(
            (set(item) == {"source", "value"} for item in result.evidence.values())
        )

    def test_missing_signals_are_unknown(self):
        result = FailureDiagnoser().diagnose(
            DiagnosisSignals(scoreable=True, cell_input={"scoreable": True})
        )
        assert not result.sufficient
        assert result.failure_family == "unknown"
        assert not result.retryable
        assert result.evidence["rule_id"]["value"] is None

    def test_cell_record_adapter_remains_explicit(self):
        signals = DiagnosisSignals.from_cell_record_input(
            {"predicate": False, "scored_reason": "agent_gave_up"}
        )
        assert signals.scoreable
        assert not signals.libero_predicate
        with pytest.raises(ValueError):
            DiagnosisSignals.from_cell_record_input(None)

    def test_protocol_end_reasons_remain_scored_diagnoses(self):
        for reason in ("budget_exhausted", "agent_gave_up", "no_progress"):
            result = FailureDiagnoser().diagnose(
                {"predicate": False, "scored_reason": reason}, scoreable=True
            )
            assert result.termination_reason == reason
            assert result.failure_family == reason
            assert result.evidence["rule_id"]["value"] == "scored_end_reason"
        with pytest.raises(ValueError, match="unexpected scored_reason"):
            FailureDiagnoser().diagnose(
                {"predicate": False, "scored_reason": "typo"}, scoreable=True
            )

    def test_missing_capability_does_not_invent_world_evidence(self):
        result = FailureDiagnoser().diagnose(
            DiagnosisSignals(scoreable=True, missing_capability="insert")
        )
        assert result.failure_family == "missing_capability"
        assert result.required_capability == "insert"
        assert not result.world_state_invalidated
        assert not result.parameter_issue
        assert not result.retryable
        assert result.evidence["rule_id"]["value"] == "registry_missing_capability"
        event = result.to_failure_event(
            FailureEvent(episode_id="ep", goal="insert", step=0)
        )
        decision = FailureRouter().route(event, available_tool_ids=("insert",))
        assert decision.level == RecoveryLevel.UNKNOWN
        assert (
            "bank/registry version mismatch" in decision.evidence["integrity_warning"]
        )
        with_world = FailureEvent(
            episode_id="ep",
            goal="insert",
            step=0,
            required_capability="insert",
            world_state_invalidated=True,
        )
        assert (
            FailureRouter().route(with_world, available_tool_ids=("insert",)).level
            == RecoveryLevel.UNKNOWN
        )

    def test_objective_evidence_precedes_heuristic_and_self_report_cannot_route(self):
        diagnoser = FailureDiagnoser()
        result = diagnoser.diagnose(
            DiagnosisSignals(
                scoreable=True,
                pi05_heuristic={
                    "failure_family": "grasp_contact",
                    "retryable": True,
                    "world_state_invalidated": True,
                },
                sam3_observation={"blocked": True},
            )
        )
        assert result.failure_family == "world_state_invalidated"
        assert result.evidence["rule_id"]["value"] == "sam3_world_state"
        assert not result.retryable
        assert result.world_state_invalidated
        uncorroborated = diagnoser.diagnose(
            DiagnosisSignals(
                scoreable=True,
                pi05_heuristic={
                    "failure_family": "grasp_contact",
                    "parameter_issue": True,
                    "retryable": True,
                },
            )
        )
        assert uncorroborated.evidence["rule_id"]["value"] == "heuristic_failure_family"
        assert not uncorroborated.parameter_issue
        assert not uncorroborated.retryable

    def test_confusion_matrix_rejects_duplicates_and_counts_fixture(self):
        fixture = Path(__file__).parent / "fixtures" / "diagnoser_audit.jsonl"
        labels = fixture
        with TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            records = []
            for line in fixture.read_text().splitlines():
                row = json.loads(line)
                assert row["provenance"] == "synthetic"
                signals = DiagnosisSignals(
                    scoreable=row["scoreable"],
                    cell_input={"scoreable": row["scoreable"]},
                    **{
                        key: row[key]
                        for key in (
                            "libero_predicate",
                            "pi05_heuristic",
                            "sam3_observation",
                            "end_effector_pose",
                            "gripper_opening",
                            "transcript_text",
                        )
                    },
                    tool_error=row.get("tool_error"),
                    missing_capability=row.get("missing_capability"),
                )
                result = FailureDiagnoser().diagnose(signals)
                records.append(
                    {
                        "event_id": row["event_id"],
                        "scoreable": True,
                        "failure_family": result.failure_family,
                    }
                )
            predictions.write_text(
                "".join((json.dumps(record) + "\n" for record in records))
            )
            matrix = diagnoser_confusion_matrix(predictions, labels)
            predictions.write_text(
                predictions.read_text() + json.dumps(records[0]) + "\n"
            )
            with pytest.raises(ValueError, match="duplicate diagnosis event_id"):
                diagnoser_confusion_matrix(predictions, labels)
        assert sum((sum(row.values()) for row in matrix.values())) == 20
        assert matrix["unknown"] == {"unknown": 1}
