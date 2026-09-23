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
    CandidateTool,
    MockToolSynthesizer,
    RPentHandoff,
    ToolGapEvent,
    ToolRegistry,
    ToolSpec,
    ToolSynthesisCoordinator,
    ToolVerifier,
    TransportToolSynthesizer,
)


class TestEvolutionCore:
    def test_mock_capx_synthesizes_verifies_and_registers_tool(self):
        failure = ToolGapEvent(
            episode_id="ep-capx",
            goal="insert peg",
            step=1,
            state={},
            failure_family="blocked_insertion",
            diagnosis={},
            required_capability="insert_around_blocker",
            missing_capability="insert_around_blocker",
        )
        handoff = RPentHandoff.from_failure(failure)
        registry = ToolRegistry()
        coordinator = ToolSynthesisCoordinator(registry, ToolVerifier())
        result = coordinator.synthesize_and_register(
            handoff,
            MockToolSynthesizer(),
            [({}, {})],
            postcondition=lambda output, context: output.get("solved") is True,
        )
        assert result.registered
        assert result.verification.passed
        assert registry.has("capx.insert_around_blocker")

    def test_failed_candidate_is_not_registered(self):
        failure = ToolGapEvent(
            episode_id="ep-capx-fail",
            goal="insert peg",
            step=1,
            state={},
            failure_family="blocked_insertion",
            diagnosis={},
            required_capability="bad_insert",
            missing_capability="bad_insert",
        )
        handoff = RPentHandoff.from_failure(failure)
        registry = ToolRegistry()
        coordinator = ToolSynthesisCoordinator(registry, ToolVerifier())
        result = coordinator.synthesize_and_register(
            handoff,
            MockToolSynthesizer({"bad_insert": lambda args, ctx: {"solved": False}}),
            [({}, {})],
            postcondition=lambda output, context: output.get("solved") is True,
        )
        assert not result.registered
        assert not registry.has("capx.bad_insert")

    def test_transport_synthesizer_requires_tool_decoder_and_rejects_skill_response(
        self,
    ):
        failure = ToolGapEvent(
            episode_id="ep-transport",
            goal="insert",
            step=1,
            failure_family="blocked",
            diagnosis={},
            required_capability="insert",
            missing_capability="insert",
        )
        handoff = RPentHandoff.from_failure(failure)
        seen = {}

        def transport(payload):
            seen.update(payload)
            return {"kind": "tool", "tool_id": "remote.insert"}

        def decoder(response, received_handoff):
            assert response["tool_id"] == "remote.insert"
            assert received_handoff.episode_id == "ep-transport"
            return CandidateTool(
                ToolSpec("remote.insert", "Remote insert", "sandbox handle"),
                lambda args, ctx: {"solved": True},
            )

        synthesizer = TransportToolSynthesizer(transport, decoder)
        candidate = synthesizer.synthesize(handoff)
        assert candidate.spec.tool_id == "remote.insert"
        assert seen["schema"] == "agentic-em/rpent-tool-gap/v1"
        with pytest.raises(RuntimeError):
            TransportToolSynthesizer(
                lambda payload: {"kind": "skill"}, decoder
            ).synthesize(handoff)
