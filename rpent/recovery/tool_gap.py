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

"""Narrow RPent boundary adapter for the tool-gap handoff.

This adapter does not implement planning.  It is deliberately limited to the
contract that a real RPent runner must satisfy before a remote backend is
connected.
"""

from __future__ import annotations

from typing import Any, Mapping

from .events import FailureEvent, RecoveryAction
from .handoff import RPentHandoff
from .router import FailureRouter
from .synthesis import SynthesisResult, ToolSynthesisCoordinator, ToolSynthesizer
from .tools import ToolRegistry
from .verification import Case, Check


class ToolGapAdapter:
    """Convert an L3 event into a safe handoff and dispatch synthesis."""

    def __init__(
        self, registry: ToolRegistry, router: FailureRouter | None = None
    ) -> None:
        """Bind the target registry and optional routing policy.

        Args:
            registry: Registry receiving verified synthesized tools.
            router: Optional failure router override.
        """
        self.registry = registry
        self.router = router or FailureRouter()

    def make_handoff(
        self,
        event: FailureEvent,
        *,
        constraints: Mapping[str, Any] | None = None,
        resource_budget: Mapping[str, Any] | None = None,
    ) -> RPentHandoff:
        """Validate an L3 event and build its descriptor-only handoff.

        Args:
            event: Failure expected to route to tool synthesis.
            constraints: Optional synthesis constraints.
            resource_budget: Optional synthesis resource limits.

        Returns:
            Handoff containing failure state and available tool schemas.

        Raises:
            ValueError: If the event does not route to L3 synthesis.
        """
        decision = self.router.route(
            event,
            available_tool_ids=(spec.tool_id for spec in self.registry.list_specs()),
        )
        if decision.action is not RecoveryAction.SYNTHESIZE_TOOL:
            raise ValueError(
                f"event does not represent an L3 tool gap: {decision.action.value}"
            )
        return RPentHandoff.from_failure(
            event,
            available_tool_schemas=tuple(
                spec.to_manifest() for spec in self.registry.list_specs()
            ),
            constraints=constraints,
            resource_budget=resource_budget,
        )

    def synthesize_tool_gap(
        self,
        event: FailureEvent,
        synthesizer: ToolSynthesizer,
        cases: list[Case] | tuple[Case, ...],
        *,
        constraints: Mapping[str, Any] | None = None,
        resource_budget: Mapping[str, Any] | None = None,
        precondition: Check | None = None,
        postcondition: Check | None = None,
    ) -> SynthesisResult:
        """Synthesize, verify, and conditionally register an L3 tool.

        Args:
            event: Failure expected to route to tool synthesis.
            synthesizer: Candidate synthesis backend.
            cases: Verification argument and context pairs.
            constraints: Optional synthesis constraints.
            resource_budget: Optional synthesis resource limits.
            precondition: Optional verification precondition.
            postcondition: Optional verification postcondition.

        Returns:
            Candidate, verification report, and registration status.
        """
        handoff = self.make_handoff(
            event,
            constraints=constraints,
            resource_budget=resource_budget,
        )
        return ToolSynthesisCoordinator(self.registry).synthesize_and_register(
            handoff,
            synthesizer,
            cases,
            precondition=precondition,
            postcondition=postcondition,
        )
