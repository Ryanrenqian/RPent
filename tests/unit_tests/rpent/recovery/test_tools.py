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
    SkillPlaybook,
    SkillStep,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)


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

    def test_tool_registry_times_success_as_well(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec("ok", "Ok", "success"), lambda args, state: {"ok": True}
        )
        assert registry.invoke(ToolCall("ok")).metadata["wall_clock_s"] >= 0

    @pytest.mark.parametrize("description", ("", "   "))
    def test_tool_spec_requires_nonempty_description(self, description):
        with pytest.raises(ValueError, match="description must not be empty"):
            ToolSpec("invalid", "Invalid", description)
