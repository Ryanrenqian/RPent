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

"""The explicit RPent-to-tool-synthesis handoff contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .events import FailureEvent, ToolGapEvent

HANDOFF_SCHEMA = "agentic-em/rpent-tool-gap/v1"


@dataclass(frozen=True, slots=True)
class RPentHandoff:
    """State and constraints passed across the RPent/CaP-X boundary.

    Only descriptors and schemas cross the boundary.  Executor objects and
    source code stay on the tool side of the runtime.
    """

    episode_id: str
    goal: str
    state: Mapping[str, Any]
    completed_subgoals: tuple[str, ...]
    failed_tool_id: str | None
    failure_family: str
    diagnosis: Mapping[str, Any]
    required_capability: str
    constraints: Mapping[str, Any] = field(default_factory=dict)
    available_tool_schemas: tuple[Mapping[str, Any], ...] = ()
    resource_budget: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields and copy all handoff containers."""
        for name in ("episode_id", "goal", "failure_family", "required_capability"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not self.required_capability.strip():
            raise ValueError("required_capability must not be empty")
        object.__setattr__(self, "state", dict(self.state))
        object.__setattr__(self, "diagnosis", dict(self.diagnosis))
        object.__setattr__(self, "constraints", dict(self.constraints))
        object.__setattr__(self, "resource_budget", dict(self.resource_budget))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "completed_subgoals", tuple(self.completed_subgoals))
        object.__setattr__(
            self,
            "available_tool_schemas",
            tuple(dict(item) for item in self.available_tool_schemas),
        )

    @classmethod
    def from_failure(
        cls,
        event: FailureEvent,
        *,
        available_tool_schemas: tuple[Mapping[str, Any], ...] = (),
        constraints: Mapping[str, Any] | None = None,
        resource_budget: Mapping[str, Any] | None = None,
    ) -> "RPentHandoff":
        """Create a synthesis handoff from an L3 failure.

        Args:
            event: Tool-gap failure carrying the required capability.
            available_tool_schemas: Descriptor-only schemas already available.
            constraints: Optional synthesis constraints.
            resource_budget: Optional synthesis resource budget.

        Returns:
            Validated descriptor-only handoff.

        Raises:
            ValueError: If ``event`` is not a tool gap or lacks a capability.
        """
        capability = event.required_capability
        if isinstance(event, ToolGapEvent):
            capability = event.missing_capability
        if not capability:
            raise ValueError("failure event has no required or missing capability")
        return cls(
            episode_id=event.episode_id,
            goal=event.goal,
            state=event.state,
            completed_subgoals=event.completed_subgoals,
            failed_tool_id=event.tool_id,
            failure_family=event.failure_family,
            diagnosis=event.diagnosis,
            required_capability=capability,
            constraints=constraints or {},
            available_tool_schemas=available_tool_schemas,
            resource_budget=resource_budget or {},
            metadata={"failure_step": event.step, "failure_outcome": event.outcome},
        )

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this handoff without executable objects.

        Returns:
            Schema-tagged, JSON-compatible handoff manifest.
        """
        return {
            "schema": HANDOFF_SCHEMA,
            "episode_id": self.episode_id,
            "goal": self.goal,
            "state": dict(self.state),
            "completed_subgoals": list(self.completed_subgoals),
            "failed_tool_id": self.failed_tool_id,
            "failure_family": self.failure_family,
            "diagnosis": dict(self.diagnosis),
            "required_capability": self.required_capability,
            "constraints": dict(self.constraints),
            "available_tool_schemas": [
                dict(item) for item in self.available_tool_schemas
            ],
            "resource_budget": dict(self.resource_budget),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "RPentHandoff":
        """Restore a handoff from its schema-tagged manifest.

        Args:
            payload: Serialized handoff fields.

        Returns:
            Validated handoff.

        Raises:
            ValueError: If the schema tag is unsupported.
        """
        if payload.get("schema") != HANDOFF_SCHEMA:
            raise ValueError(f"unsupported handoff schema: {payload.get('schema')!r}")
        return cls(
            episode_id=str(payload["episode_id"]),
            goal=str(payload["goal"]),
            state=dict(payload.get("state", {})),
            completed_subgoals=tuple(payload.get("completed_subgoals", ())),
            failed_tool_id=payload.get("failed_tool_id"),
            failure_family=str(payload["failure_family"]),
            diagnosis=dict(payload.get("diagnosis", {})),
            required_capability=str(payload["required_capability"]),
            constraints=dict(payload.get("constraints", {})),
            available_tool_schemas=tuple(payload.get("available_tool_schemas", ())),
            resource_budget=dict(payload.get("resource_budget", {})),
            metadata=dict(payload.get("metadata", {})),
        )
