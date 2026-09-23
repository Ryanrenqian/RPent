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

import pytest

from rpent.recovery import (
    DiagnosisResult,
    DiagnosisSignals,
    FailureDiagnoser,
    SkillPlaybook,
    SkillStep,
)
from rpent.recovery.adapt import ParameterAdapter


class TestBudgetAndRuntime:
    @staticmethod
    def skill(*tool_ids):
        return SkillPlaybook(
            skill_id="timed",
            name="Timed",
            goal="pick",
            trigger_labels=frozenset({"x"}),
            diagnosis="d",
            steps=tuple(
                (SkillStep(str(i), "p", tool_id) for i, tool_id in enumerate(tool_ids))
            ),
            verification_checks=("v",),
        )

    def test_parameter_adapter_requires_explicit_numeric_target(self):
        pose = FailureDiagnoser().diagnose(
            DiagnosisSignals(
                end_effector_pose={"reachable": False, "target_pose": [1, 2, 3]},
            )
        )
        adapted = ParameterAdapter.adapt(pose, {"pose": [0, 0, 0]})
        assert adapted.delta == {"pose": [1, 2, 3]}
        no_target = FailureDiagnoser().diagnose(
            DiagnosisSignals(end_effector_pose={"reachable": False})
        )
        assert not ParameterAdapter.adapt(
            no_target, {"pose": [0, 0, 0]}
        ).adaptation_available

    @pytest.mark.parametrize(
        "evidence",
        (
            {
                "end_effector_pose": {
                    "source": "test",
                    "value": {"reachable": False, "target_pose": [1, 2, 3]},
                }
            },
            {
                "gripper_opening": {
                    "source": "test",
                    "value": {"value": 0.0, "target_opening": 0.4},
                },
                "transcript_text": {"source": "test", "value": "grasp failed"},
            },
            {
                "gripper_opening": {"source": "test", "value": {"value": 0.0}},
                "transcript_text": {"source": "test", "value": "grasp failed"},
                "approach_distance": {
                    "source": "test",
                    "value": {"target": 0.2},
                },
            },
        ),
    )
    def test_parameter_adapter_does_not_invent_argument_names(self, evidence):
        diagnosis = DiagnosisResult(
            "parameter_or_pose",
            evidence,
            retryable=False,
            parameter_issue=True,
            world_state_invalidated=False,
            required_capability=None,
            sufficient=True,
        )

        adaptation = ParameterAdapter.adapt(diagnosis, {"other": 7})

        assert adaptation.delta == {}
        assert adaptation.adaptation_available is False
