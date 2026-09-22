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
    OutcomeVerifier,
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    SkillValidation,
    SkillVerifier,
    ToolRegistry,
    ToolSpec,
    ToolVerifier,
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
    def test_tool_verifier_requires_cases_and_postcondition(self):
        spec = ToolSpec("restage-tool", "Restage", "move object", version="2")
        verifier = ToolVerifier()
        report = verifier.verify(
            spec,
            lambda args, state: {"reachable": True},
            [({"object": "cup"}, {})],
            postcondition=lambda output, state: output.get("reachable") is True,
        )
        assert report.passed
        failed = verifier.verify(
            spec,
            lambda args, state: {"reachable": False},
            [({}, {})],
            postcondition=lambda output, state: output.get("reachable") is True,
        )
        assert not failed.passed
        registry = ToolRegistry()
        registry.register_verified(
            spec, lambda args, state: {"reachable": True}, report
        )
        with pytest.raises(RuntimeError):
            registry.register_verified(
                spec, lambda args, state: {"reachable": True}, failed
            )

    def test_outcome_and_skill_verifiers_have_separate_gates(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec("restage-tool", "Restage", "move object"),
            lambda args, state: {"reachable": True},
        )
        result = SkillRuntime(registry).execute(validated(), episode_id="ep-5")
        outcome = OutcomeVerifier().verify(
            result, {"reachable": lambda r: r.tool_results[-1].output["reachable"]}
        )
        assert outcome.passed
        report = SkillVerifier().verify(
            validated(),
            {
                "replay": lambda skill: True,
                "perturbation": lambda skill: True,
                "held_out": lambda skill: False,
                "regression": lambda skill: True,
            },
        )
        assert not report.passed
        assert "held_out" in report.failures
