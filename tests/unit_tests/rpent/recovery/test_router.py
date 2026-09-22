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

from rpent.recovery import (
    FailureEvent,
    FailureRouter,
    RecoveryAction,
    RecoveryLevel,
    SkillPlaybook,
    SkillStep,
    SkillValidation,
    ToolGapEvent,
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


class TestEvolutionCore:
    def test_router_prioritizes_tool_gap_then_world_then_parameter_then_retry(self):
        router = FailureRouter()
        common = {
            "episode_id": "ep",
            "goal": "pick",
            "step": 1,
            "failure_family": "x",
        }
        assert (
            router.route(ToolGapEvent(**common, missing_capability="insert")).level
            == RecoveryLevel.L3
        )
        assert (
            router.route(FailureEvent(**common, world_state_invalidated=True)).action
            == RecoveryAction.REPLAN
        )
        assert (
            router.route(FailureEvent(**common, parameter_issue=True)).action
            == RecoveryAction.ADAPT
        )
        assert (
            router.route(FailureEvent(**common, retryable=True)).action
            == RecoveryAction.RETRY
        )
        assert router.oracle(RecoveryLevel.L3).source == "oracle"


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

    def test_unknown_router_action_and_oracle(self):
        event = FailureEvent(
            episode_id="ep", goal="pick", step=0, failure_family="unknown"
        )
        router = FailureRouter()
        assert router.route(event).level == RecoveryLevel.UNKNOWN
        assert router.oracle(RecoveryLevel.UNKNOWN).action == RecoveryAction.GIVE_UP
