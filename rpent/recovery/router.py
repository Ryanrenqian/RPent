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

"""Auditable L0-L3 routing rules."""

from __future__ import annotations

from typing import Iterable

from .events import (
    FailureEvent,
    RecoveryAction,
    RecoveryDecision,
    RecoveryLevel,
    ToolGapEvent,
)


class FailureRouter:
    """Route failures by explicit evidence, with stable priority ordering."""

    def route(
        self, event: FailureEvent, *, available_tool_ids: Iterable[str] = ()
    ) -> RecoveryDecision:
        """Route a failure using fixed L3-to-L0 precedence.

        Args:
            event: Diagnosed execution failure.
            available_tool_ids: Tool IDs currently registered.

        Returns:
            Auditable rule-based recovery decision.
        """
        available = frozenset(available_tool_ids)
        if (
            isinstance(event, ToolGapEvent)
            or event.required_capability
            and event.required_capability not in available
        ):
            capability = event.required_capability or getattr(
                event, "missing_capability", "unknown"
            )
            return RecoveryDecision(
                RecoveryLevel.L3,
                RecoveryAction.SYNTHESIZE_TOOL,
                "no registered tool satisfies the required capability",
                {
                    "required_capability": capability,
                    "available_tools": sorted(available),
                },
            )
        if event.required_capability and event.required_capability in available:
            return RecoveryDecision(
                RecoveryLevel.UNKNOWN,
                RecoveryAction.GIVE_UP,
                "claimed missing capability is registered; possible bank/registry version mismatch",
                {
                    "required_capability": event.required_capability,
                    "available_tools": sorted(available),
                    "integrity_warning": "capability claimed missing but registry contains it; possible bank/registry version mismatch",
                },
            )
        if event.world_state_invalidated:
            return RecoveryDecision(
                RecoveryLevel.L2,
                RecoveryAction.REPLAN,
                "the observed world state invalidated the current plan",
                {"failure_family": event.failure_family},
            )
        if event.parameter_issue:
            return RecoveryDecision(
                RecoveryLevel.L1,
                RecoveryAction.ADAPT,
                "the selected tool is applicable but its parameters or pose are wrong",
                {"diagnosis": dict(event.diagnosis)},
            )
        if event.retryable:
            return RecoveryDecision(
                RecoveryLevel.L0,
                RecoveryAction.RETRY,
                "the failure is marked as a bounded stochastic execution miss",
                {"failure_family": event.failure_family},
            )
        return RecoveryDecision(
            RecoveryLevel.UNKNOWN,
            RecoveryAction.GIVE_UP,
            "failure lacks sufficient evidence for a recovery route",
            {"failure_family": event.failure_family, "evidence_sufficient": False},
        )

    def oracle(
        self, level: RecoveryLevel, *, reason: str = "oracle label"
    ) -> RecoveryDecision:
        """Create an oracle-labelled decision for offline comparison.

        Args:
            level: Expected recovery level.
            reason: Explanation attached to the oracle label.

        Returns:
            Oracle-sourced decision with the action for ``level``.
        """
        action = {
            RecoveryLevel.L0: RecoveryAction.RETRY,
            RecoveryLevel.L1: RecoveryAction.ADAPT,
            RecoveryLevel.L2: RecoveryAction.REPLAN,
            RecoveryLevel.L3: RecoveryAction.SYNTHESIZE_TOOL,
            RecoveryLevel.UNKNOWN: RecoveryAction.GIVE_UP,
        }[level]
        return RecoveryDecision(level, action, reason, source="oracle")
