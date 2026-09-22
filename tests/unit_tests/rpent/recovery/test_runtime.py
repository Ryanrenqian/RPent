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

import time
from operator import eq

from rpent.recovery import (
    BudgetLedger,
    BudgetLimits,
    DiagnosisResult,
    RecoveryAction,
    RecoveryLevel,
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    SkillValidation,
    ToolGapEvent,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from rpent.recovery.budget import PLANNER_TIMEOUT_S


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
    def test_runtime_calls_tools_and_does_not_need_tool_source(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "restage-tool", "Restage", "move object", postconditions=("reachable",)
            ),
            lambda args, state: {"reachable": True},
        )
        result = SkillRuntime(registry).execute(validated(), episode_id="ep-2")
        assert result.success
        assert eq(result.tool_results[0].output["reachable"], True)
        assert result.events[-1].outcome == "succeeded"

    def test_runtime_turns_tool_exception_into_failure_and_route(self):
        registry = ToolRegistry()
        registry.register(ToolSpec("bad", "Bad", "fails"), lambda args, state: 1 / 0)
        skill = SkillPlaybook(
            skill_id="bad-skill",
            name="Bad",
            goal="pick",
            trigger_labels=frozenset({"x"}),
            diagnosis="d",
            steps=(SkillStep("s", "p", "bad"),),
            verification_checks=("v",),
            validation=SkillValidation("e", True, True, True, True),
        )
        result = SkillRuntime(registry).execute(skill, episode_id="ep-3")
        assert not result.success
        assert result.decision.action == RecoveryAction.RETRY

    def test_missing_tool_becomes_l3_tool_gap(self):
        result = SkillRuntime(ToolRegistry()).execute(validated(), episode_id="ep-4")
        assert not result.success
        assert result.decision.level == RecoveryLevel.L3
        assert result.decision.action == RecoveryAction.SYNTHESIZE_TOOL
        assert isinstance(result.events[-1], ToolGapEvent)
        assert result.events[-1].missing_capability == "restage-tool"


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

    def test_runtime_clock_does_not_depend_on_tool_metadata(self):

        class MetadataFreeRegistry(ToolRegistry):
            def invoke(self, call, context=None):
                original = super().invoke(call, context)
                return ToolResult(
                    call.tool_id, original.success, original.output, original.error
                )

        registry = MetadataFreeRegistry()

        def slow_success(args, state):
            time.sleep(0.02)
            return {"done": True}

        registry.register(ToolSpec("slow", "Slow", "success"), slow_success)
        result = SkillRuntime(
            registry, ledger=BudgetLedger(limits=BudgetLimits(max_wall_clock_s=0.01))
        ).execute(self.skill("slow", "slow"), episode_id="ep-no-metadata")
        assert result.tool_results[0].metadata == {}
        assert result.decision.termination_reason == "budget_exhausted"
        assert result.ledger.wall_clock_s > 0.01

    def test_l1_pose_adaptation_changes_second_call_arguments(self):
        calls = []

        class EvidenceRegistry(ToolRegistry):
            def invoke(self, call, context=None):
                calls.append(dict(call.arguments))
                if len(calls) == 1:
                    return ToolResult(
                        call.tool_id,
                        False,
                        error="bad pose",
                        metadata={
                            "end_effector_pose": {
                                "reachable": False,
                                "target_pose": [1, 2, 3],
                            }
                        },
                    )
                return ToolResult(call.tool_id, True, output={"done": True})

        registry = EvidenceRegistry()
        registry.register(ToolSpec("move", "Move", "move"), lambda args, state: {})
        skill = self.skill("move")
        skill = SkillPlaybook(
            skill.skill_id,
            skill.name,
            skill.goal,
            skill.trigger_labels,
            skill.diagnosis,
            (SkillStep("0", "p", "move", {"pose": [0, 0, 0]}),),
            skill.verification_checks,
        )
        result = SkillRuntime(registry, recovery_loop=True).execute(
            skill, episode_id="ep-pose"
        )
        assert result.success
        assert calls == [{"pose": [0, 0, 0]}, {"pose": [1, 2, 3]}]

    def test_l1_gripper_adaptation_changes_second_call_arguments(self):
        calls = []

        class EvidenceRegistry(ToolRegistry):
            def invoke(self, call, context=None):
                calls.append(dict(call.arguments))
                if len(calls) == 1:
                    return ToolResult(
                        call.tool_id,
                        False,
                        error="no grasp",
                        metadata={
                            "gripper_opening": {"value": 0.0, "target_opening": 0.4},
                            "transcript_text": "grasp failed",
                        },
                    )
                return ToolResult(call.tool_id, True, output={"done": True})

        registry = EvidenceRegistry()
        registry.register(ToolSpec("grasp", "Grasp", "grasp"), lambda args, state: {})
        skill = SkillPlaybook(
            "grasp",
            "Grasp",
            "pick",
            frozenset({"x"}),
            "d",
            (SkillStep("0", "p", "grasp", {"gripper_opening": 0.0}),),
            ("v",),
        )
        result = SkillRuntime(registry, recovery_loop=True).execute(
            skill, episode_id="ep-gripper"
        )
        assert result.success
        assert calls == [{"gripper_opening": 0.0}, {"gripper_opening": 0.4}]

    def test_l1_without_evidence_gives_up_instead_of_raw_retry(self):

        class L1WithoutEvidence:
            def diagnose(self, signals):
                return DiagnosisResult(
                    "parameter_or_pose",
                    {
                        "evidence_sufficiency": {"source": "test", "value": True},
                        "end_effector_pose": {
                            "source": "test",
                            "value": {"reachable": False},
                        },
                    },
                    False,
                    True,
                    False,
                    None,
                    True,
                )

        calls = []
        registry = ToolRegistry()
        registry.register(
            ToolSpec("bad", "Bad", "bad"),
            lambda args, state: calls.append(dict(args)) or 1 / 0,
        )
        result = SkillRuntime(
            registry, diagnoser=L1WithoutEvidence(), recovery_loop=True
        ).execute(
            SkillPlaybook(
                "l1",
                "L1",
                "pick",
                frozenset({"x"}),
                "d",
                (SkillStep("0", "p", "bad", {"pose": [0, 0, 0]}),),
                ("v",),
            ),
            episode_id="ep-no-evidence",
        )
        assert not result.success
        assert calls == [{"pose": [0, 0, 0]}]
        assert "L1 无可用自适应证据" in result.decision.reason

    def test_l1_adaptation_has_finite_termination_without_planner_timeout(self):

        class AlwaysPoseFailure(ToolRegistry):
            def invoke(self, call, context=None):
                return ToolResult(
                    call.tool_id,
                    False,
                    error="still bad",
                    metadata={
                        "end_effector_pose": {
                            "reachable": False,
                            "target_pose": [1, 2, 3],
                        }
                    },
                )

        registry = AlwaysPoseFailure()
        registry.register(ToolSpec("bad", "Bad", "bad"), lambda args, state: {})
        result = SkillRuntime(registry, recovery_loop=True).execute(
            SkillPlaybook(
                "l1-loop",
                "L1 loop",
                "pick",
                frozenset({"x"}),
                "d",
                (SkillStep("0", "p", "bad", {"pose": [0, 0, 0]}),),
                ("v",),
            ),
            episode_id="ep-l1-loop",
        )
        assert not result.success
        assert (
            result.decision.termination_reason
            == "L1 adaptation produced no argument change"
        )
        assert result.ledger.attempts == 2
        assert result.ledger.parameter_adaptations == {"pose": 2}
        assert result.ledger.wall_clock_s < PLANNER_TIMEOUT_S

    def test_parameter_adaptation_limit_is_binding_when_lower_than_no_progress(self):

        class ChangingPoseFailure(ToolRegistry):
            def invoke(self, call, context=None):
                next_x = call.arguments["pose"][0] + 1
                return ToolResult(
                    call.tool_id,
                    False,
                    error="still bad",
                    metadata={
                        "end_effector_pose": {
                            "reachable": False,
                            "target_pose": [next_x, 0, 0],
                        }
                    },
                )

        registry = ChangingPoseFailure()
        registry.register(ToolSpec("bad", "Bad", "bad"), lambda args, state: {})
        ledger = BudgetLedger(
            limits=BudgetLimits(
                max_turns=100,
                max_wall_clock_s=4000,
                max_upper_model_calls=100,
                max_env_steps=10000,
            ),
            max_no_progress=99,
            max_parameter_adaptations=2,
        )
        result = SkillRuntime(registry, ledger=ledger, recovery_loop=True).execute(
            SkillPlaybook(
                "l1-limit",
                "L1 limit",
                "pick",
                frozenset({"x"}),
                "d",
                (SkillStep("0", "p", "bad", {"pose": [0, 0, 0]}),),
                ("v",),
            ),
            episode_id="ep-l1-limit",
        )
        assert not result.success
        assert result.decision.termination_reason == "parameter_adaptation_limit"
        assert result.ledger.attempts == 3
        assert result.ledger.parameter_adaptations == {"pose": 2}
        assert result.decision.evidence["adaptation_delta"] == {"pose": [3, 0, 0]}
        assert (
            result.decision.evidence["adaptation_evidence"]["source"] == "target_pose"
        )
        assert result.decision.evidence["adaptation_evidence"]["new"] == [3, 0, 0]

    def test_l1_failure_uses_real_registry_timing_path(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec("observe", "Observe", "records pose evidence"),
            lambda args, state: {
                "end_effector_pose": {"reachable": False, "target_pose": [1, 2, 3]}
            },
        )

        def timed_failure(args, state):
            time.sleep(0.001)
            raise RuntimeError("still bad")

        registry.register(
            ToolSpec("bad", "Bad", "fails through registry"), timed_failure
        )
        skill = SkillPlaybook(
            "real-timing",
            "Real timing",
            "pick",
            frozenset({"x"}),
            "d",
            (
                SkillStep("0", "observe", "observe"),
                SkillStep("1", "move", "bad", {"pose": [0, 0, 0]}),
            ),
            ("v",),
        )
        result = SkillRuntime(registry, recovery_loop=True).execute(
            skill, episode_id="ep-real-timing"
        )
        assert not result.success
        assert (
            result.decision.termination_reason
            == "L1 adaptation produced no argument change"
        )
        assert result.ledger.attempts == 3
        assert result.ledger.parameter_adaptations == {"pose": 2}
        assert result.ledger.wall_clock_s > 0

    def test_repeated_failed_tool_gives_up_finitely(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec("bad", "Bad", "always fails"), lambda args, state: 1 / 0
        )
        skill = SkillPlaybook(
            skill_id="loop",
            name="Loop",
            goal="pick",
            trigger_labels=frozenset({"x"}),
            diagnosis="d",
            steps=(SkillStep("s", "p", "bad"),),
            verification_checks=("v",),
        )
        ledger = BudgetLedger(max_no_progress=2)
        result = SkillRuntime(registry, ledger=ledger, recovery_loop=True).execute(
            skill, episode_id="ep-loop"
        )
        assert not result.success
        assert result.decision.action == RecoveryAction.GIVE_UP
        assert result.decision.termination_reason == "no_progress"
        assert result.ledger.attempts <= 3
        assert any(
            (
                getattr(event, "termination_reason", None) == "no_progress"
                for event in result.events
            )
        )

    def test_budget_exhaustion_after_progress_is_explicit(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec("ok", "Ok", "progress"), lambda args, state: {"done": True}
        )
        skill = SkillPlaybook(
            skill_id="budget",
            name="Budget",
            goal="pick",
            trigger_labels=frozenset({"x"}),
            diagnosis="d",
            steps=(SkillStep("a", "a", "ok"), SkillStep("b", "b", "ok")),
            verification_checks=("v",),
        )
        result = SkillRuntime(
            registry, ledger=BudgetLedger(limits=BudgetLimits(max_env_steps=1))
        ).execute(skill, episode_id="ep-budget")
        assert not result.success
        assert result.decision.termination_reason == "budget_exhausted"
        assert result.decision.action == RecoveryAction.GIVE_UP
        assert result.decision.evidence["evidence_sufficient"]

    def test_give_up_preserves_insufficient_diagnosis(self):

        class InsufficientDiagnoser:
            def diagnose(self, signals):
                return DiagnosisResult(
                    "unknown",
                    {"evidence_sufficiency": {"source": "test", "value": False}},
                    True,
                    False,
                    False,
                    None,
                    False,
                )

        registry = ToolRegistry()
        registry.register(ToolSpec("bad", "Bad", "failure"), lambda args, state: 1 / 0)
        result = SkillRuntime(
            registry,
            diagnoser=InsufficientDiagnoser(),
            ledger=BudgetLedger(limits=BudgetLimits(max_env_steps=1)),
            recovery_loop=True,
        ).execute(self.skill("bad"), episode_id="ep-insufficient")
        assert result.decision.termination_reason == "budget_exhausted"
        assert not result.decision.evidence["evidence_sufficient"]
