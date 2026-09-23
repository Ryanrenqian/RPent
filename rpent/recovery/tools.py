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

"""Tool contracts and an in-process registry for the first runtime slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from .persistence import TOOL_MANIFEST_SCHEMA, read_manifest, write_manifest

ToolExecutor = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Describes an executable state-transition operator, not its source."""

    tool_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    resource_budget: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"
    source: str = "builtin"

    def __post_init__(self) -> None:
        """Validate descriptor identifiers and copy container inputs."""
        for field_name in ("tool_id", "name", "description", "version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        object.__setattr__(self, "input_schema", dict(self.input_schema))
        object.__setattr__(self, "output_schema", dict(self.output_schema))
        object.__setattr__(self, "resource_budget", dict(self.resource_budget))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "postconditions", tuple(self.postconditions))
        object.__setattr__(self, "constraints", tuple(self.constraints))

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this descriptor without executable source.

        Returns:
            JSON-compatible tool descriptor.
        """
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "constraints": list(self.constraints),
            "resource_budget": dict(self.resource_budget),
            "version": self.version,
            "source": self.source,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ToolSpec":
        """Restore a tool descriptor from a manifest.

        Args:
            payload: Serialized descriptor fields.

        Returns:
            Validated tool descriptor.
        """
        return cls(
            tool_id=str(payload["tool_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            input_schema=dict(payload.get("input_schema", {})),
            output_schema=dict(payload.get("output_schema", {})),
            preconditions=tuple(payload.get("preconditions", ())),
            postconditions=tuple(payload.get("postconditions", ())),
            constraints=tuple(payload.get("constraints", ())),
            resource_budget=dict(payload.get("resource_budget", {})),
            version=str(payload.get("version", "1")),
            source=str(payload.get("source", "manifest")),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Immutable tool identifier, arguments, and optional call budget."""

    tool_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the tool ID and copy mapping inputs."""
        if not self.tool_id.strip():
            raise ValueError("tool_id must not be empty")
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "budget", dict(self.budget))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized tool outcome with output, error, and execution metadata."""

    tool_id: str
    success: bool
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy mappings and enforce success/error consistency."""
        object.__setattr__(self, "output", dict(self.output))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.success and self.error is not None:
            raise ValueError("successful ToolResult cannot carry an error")
        if not self.success and not (self.error or "").strip():
            raise ValueError("failed ToolResult must carry an error")


class ToolError(RuntimeError):
    """Raised for registry and executor failures."""


class ToolRegistry:
    """Versioned registry exposing tools by ID, never raw source to skills."""

    def __init__(self) -> None:
        """Initialize an empty descriptor and executor registry."""
        self._specs: dict[str, ToolSpec] = {}
        self._executors: dict[str, ToolExecutor] = {}

    def register(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        """Register a new descriptor and executor.

        Args:
            spec: Tool descriptor with a unique ID.
            executor: Callable implementing the descriptor.

        Raises:
            TypeError: If ``executor`` is not callable.
            ToolError: If the tool ID is already registered.
        """
        if not callable(executor):
            raise TypeError("executor must be callable")
        if spec.tool_id in self._specs:
            raise ToolError(f"tool already registered: {spec.tool_id}")
        self._specs[spec.tool_id] = spec
        self._executors[spec.tool_id] = executor

    def replace(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        """Replace an existing tool with the same ID.

        Args:
            spec: Replacement descriptor.
            executor: Replacement implementation.

        Raises:
            TypeError: If ``executor`` is not callable.
            ToolError: If the tool ID is unknown.
        """
        if not callable(executor):
            raise TypeError("executor must be callable")
        if spec.tool_id not in self._specs:
            raise ToolError(f"cannot replace unknown tool: {spec.tool_id}")
        self._specs[spec.tool_id] = spec
        self._executors[spec.tool_id] = executor

    def register_verified(
        self, spec: ToolSpec, executor: ToolExecutor, report: Any
    ) -> None:
        """Register a candidate only after an external verifier gate.

        Args:
            spec: Verified candidate descriptor.
            executor: Candidate implementation.
            report: Verification report belonging to ``spec``.

        Raises:
            ToolError: If the report does not match or did not pass.
        """
        if getattr(report, "subject_id", None) != spec.tool_id:
            raise ToolError("verification report does not belong to this tool")
        if not getattr(report, "passed", False):
            raise ToolError(f"tool did not pass verification: {spec.tool_id}")
        self.register(spec, executor)

    def get(self, tool_id: str) -> ToolSpec:
        """Return a descriptor by tool ID.

        Args:
            tool_id: Registered tool identifier.

        Returns:
            Matching tool descriptor.

        Raises:
            ToolError: If the identifier is unknown.
        """
        try:
            return self._specs[tool_id]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {tool_id}") from exc

    def has(self, tool_id: str) -> bool:
        """Return whether a tool ID is registered."""
        return tool_id in self._specs

    def list_specs(self) -> tuple[ToolSpec, ...]:
        """Return descriptors in insertion order."""
        return tuple(self._specs.values())

    def manifest(self) -> dict[str, Any]:
        """Serialize all descriptors to a schema-tagged manifest."""
        return {
            "schema": TOOL_MANIFEST_SCHEMA,
            "tools": [spec.to_manifest() for spec in self._specs.values()],
        }

    def save_manifest(self, path: str | Path) -> None:
        """Persist all descriptors atomically.

        Args:
            path: Destination manifest path.
        """
        write_manifest(path, self.manifest())

    @staticmethod
    def load_specs(path: str | Path) -> tuple[ToolSpec, ...]:
        """Load descriptors from a tool manifest.

        Args:
            path: Source manifest path.

        Returns:
            Reconstructed descriptors in manifest order.

        Raises:
            ValueError: If the manifest's ``tools`` field is not a list.
        """
        payload = read_manifest(path, TOOL_MANIFEST_SCHEMA)
        tools = payload.get("tools")
        if not isinstance(tools, list):
            raise ValueError("tool manifest 'tools' must be a list")
        return tuple(ToolSpec.from_manifest(item) for item in tools)

    def invoke(
        self, call: ToolCall, context: Mapping[str, Any] | None = None
    ) -> ToolResult:
        """Invoke a registered executor and normalize exceptions as failures.

        Args:
            call: Tool identifier and argument mapping.
            context: Optional runtime state visible to the executor.

        Returns:
            Successful mapping output or a failed result with measured duration.
        """
        self.get(call.tool_id)
        executor = self._executors[call.tool_id]
        start = monotonic()
        try:
            output = executor(call.arguments, dict(context or {}))
            if not isinstance(output, Mapping):
                raise TypeError("tool executor must return a mapping")
            return ToolResult(
                tool_id=call.tool_id,
                success=True,
                output=output,
                metadata={"wall_clock_s": monotonic() - start},
            )
        except Exception as exc:  # executor failures become data at the boundary
            return ToolResult(
                tool_id=call.tool_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"wall_clock_s": monotonic() - start},
            )
