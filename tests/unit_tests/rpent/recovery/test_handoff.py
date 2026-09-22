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
    RPentHandoff,
    ToolGapEvent,
)


class TestEvolutionCore:
    def test_rpent_handoff_round_trip_contains_state_and_tool_schemas_only(self):
        failure = ToolGapEvent(
            episode_id="ep-gap",
            goal="insert peg",
            step=4,
            state={"peg_visible": True},
            completed_subgoals=("pick",),
            tool_id="insert-tool",
            failure_family="blocked_insertion",
            diagnosis={"opening": "occluded"},
            required_capability="insert_around_blocker",
            missing_capability="insert_around_blocker",
        )
        handoff = RPentHandoff.from_failure(
            failure,
            available_tool_schemas=({"tool_id": "grasp-tool", "version": "1"},),
            constraints={"max_steps": 10},
        )
        restored = RPentHandoff.from_manifest(handoff.to_manifest())
        assert restored.required_capability == "insert_around_blocker"
        assert restored.available_tool_schemas[0]["tool_id"] == "grasp-tool"
        assert "executor" not in restored.to_manifest()
