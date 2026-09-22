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
    FailureEvent,
    MockToolSynthesizer,
    ToolGapAdapter,
    ToolGapEvent,
    ToolRegistry,
)


class TestEvolutionCore:
    def test_tool_gap_adapter_rejects_non_l3_and_dispatches_l3(self):
        registry = ToolRegistry()
        adapter = ToolGapAdapter(registry)
        non_gap = FailureEvent(
            episode_id="ep-non-gap",
            goal="pick",
            step=1,
            failure_family="bad_pose",
            parameter_issue=True,
        )
        with pytest.raises(ValueError):
            adapter.make_handoff(non_gap)
        gap = ToolGapEvent(
            episode_id="ep-adapter",
            goal="insert",
            step=1,
            failure_family="blocked",
            diagnosis={},
            required_capability="insert",
            missing_capability="insert",
        )
        result = adapter.synthesize_tool_gap(
            gap,
            MockToolSynthesizer(),
            [({}, {})],
            postcondition=lambda output, context: output.get("solved") is True,
        )
        assert result.registered
        assert registry.has("capx.insert")
