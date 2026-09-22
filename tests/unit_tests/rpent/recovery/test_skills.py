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

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rpent.recovery import (
    ExperienceSkillBank,
    SkillPlaybook,
    SkillStep,
    SkillValidation,
    ToolRegistry,
    ToolSpec,
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
    def test_skill_bank_only_promotes_verified_playbooks_and_retrieves(self):
        bank = ExperienceSkillBank()
        skill = validated()
        bank.promote(skill)
        assert bank.retrieve(goal="pick object", labels={"failed_grasp"}) is skill
        with pytest.raises(ValueError):
            bank.promote(
                SkillPlaybook(
                    skill_id="unverified",
                    name="u",
                    goal="pick",
                    trigger_labels=frozenset({"x"}),
                    diagnosis="d",
                    steps=(SkillStep("s", "p", "t"),),
                    verification_checks=("v",),
                )
            )

    def test_manifests_round_trip_descriptors_without_executors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ToolRegistry()
            registry.register(
                ToolSpec("restage-tool", "Restage", "move object"),
                lambda args, state: {"reachable": True},
            )
            tool_path = root / "tools.json"
            registry.save_manifest(tool_path)
            specs = ToolRegistry.load_specs(tool_path)
            assert ("restage-tool",) == tuple((spec.tool_id for spec in specs))
            bank = ExperienceSkillBank()
            bank.promote(validated())
            skill_path = root / "skills.json"
            bank.save_manifest(skill_path)
            restored = ExperienceSkillBank.load_manifest(skill_path)
            assert restored.get("recover-grasp").verified
            assert (
                restored.retrieve(goal="pick object", labels={"failed_grasp"}).skill_id
                == "recover-grasp"
            )
