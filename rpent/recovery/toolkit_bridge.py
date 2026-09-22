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

"""Bridge RPent toolkit execution into recovery tool contracts."""

from __future__ import annotations

from time import monotonic
from typing import Any, Mapping

from rpent.tools.toolkit import Toolkit

from .tools import ToolCall, ToolResult, ToolSpec

_IMAGE_KEYS = (
    "_image_bytes",
    "_image_cam_bytes",
    "_image_nav_bytes",
    "_image_wrist_bytes",
)


class ToolkitBackedRegistry:
    """Expose a real :class:`Toolkit` through recovery registry operations."""

    def __init__(self, toolkit: Toolkit) -> None:
        """Bind the toolkit used for descriptor lookup and execution.

        Args:
            toolkit: RPent toolkit whose registered handlers should be exposed.
        """
        self.toolkit = toolkit

    def has(self, tool_id: str) -> bool:
        """Return whether the toolkit publishes ``tool_id``.

        Args:
            tool_id: Planner-facing toolkit name.

        Returns:
            Whether a matching toolkit descriptor exists.
        """
        return any(spec.tool_id == tool_id for spec in self.list_specs())

    def list_specs(self) -> tuple[ToolSpec, ...]:
        """Convert toolkit schemas to recovery descriptors.

        Returns:
            Tool descriptors in the order published by the toolkit.
        """
        return tuple(
            ToolSpec(
                tool_id=str(schema["name"]),
                name=str(schema["name"]),
                description=str(schema["description"]),
                input_schema=dict(schema.get("input_schema", {})),
                source="rpent-toolkit",
            )
            for schema in self.toolkit.get_tools_spec()
        )

    def invoke(
        self, call: ToolCall, context: Mapping[str, Any] | None = None
    ) -> ToolResult:
        """Execute a toolkit handler and normalize its result for recovery.

        Raw image byte fields are replaced with their key names so persisted
        recovery traces remain JSON-serializable and bounded in size. Failure
        output is retained because it contains the post-call environment state
        used by L1 adaptation.

        Args:
            call: Recovery tool call containing the toolkit name and arguments.
            context: Unused registry-compatible runtime context.

        Returns:
            Normalized recovery result with measured timing and toolkit metadata.
        """
        started = monotonic()
        toolkit_result = self.toolkit.execute_tool(call.tool_id, dict(call.arguments))
        wall_clock_s = monotonic() - started

        raw_result = toolkit_result.result
        if not isinstance(raw_result, dict):
            return ToolResult(
                tool_id=call.tool_id,
                success=True,
                output={"value": raw_result},
                metadata={
                    "wall_clock_s": wall_clock_s,
                    "is_finish": toolkit_result.is_finish,
                    "call_id": toolkit_result.call_id,
                },
            )

        output = dict(raw_result)
        image_keys = [key for key in _IMAGE_KEYS if key in output]
        for key in image_keys:
            output.pop(key)
        if image_keys:
            output["_images"] = image_keys

        action_result = raw_result
        log = raw_result.get("log")
        if isinstance(log, Mapping) and isinstance(log.get("result"), Mapping):
            action_result = log["result"]
        error_value = raw_result.get("error") or action_result.get("error")
        has_error = bool(error_value)
        reported_failure = (
            raw_result.get("success") is False or action_result.get("success") is False
        )
        success = not (has_error or reported_failure)
        error = None
        if not success:
            if has_error:
                error = str(error_value)
            else:
                error = "tool result reported success=False"
        return ToolResult(
            tool_id=call.tool_id,
            success=success,
            output=output,
            error=error,
            metadata={
                "wall_clock_s": wall_clock_s,
                "is_finish": toolkit_result.is_finish,
                "call_id": toolkit_result.call_id,
            },
        )
