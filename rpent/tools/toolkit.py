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

"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the planner calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.utils.logging import get_logger, get_output_dir
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager
    from rpent.recovery.persistence import JsonlWriter
    from rpent.session import EnvState, StepRecord

logger = get_logger("toolkit")


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


def _truncate_utf8(text: str, max_bytes: int, *, marker: str = "") -> str:
    """Truncate text to a valid UTF-8 byte budget, including its marker."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    if max_bytes <= 0:
        return ""

    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) > max_bytes:
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore")
    body = encoded[: max_bytes - len(marker_bytes)].decode(
        "utf-8",
        errors="ignore",
    )
    return body + marker


def readonly(func):
    """Mark a tool handler as not advancing environment state.

    Tool handlers capture a fresh observation (:meth:`Toolkit.get_env_state`)
    by default. Apply this marker to observational and file/IO tools that do
    not move the robot or otherwise change the environment.
    """
    func._readonly = True
    return func


def _is_readonly(handler: Callable[..., Any]) -> bool:
    """Whether ``handler`` was marked with :func:`readonly`."""
    target = handler
    while isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return bool(getattr(target, "_readonly", False))


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]
    call_id: str | None = None

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    is_finish: bool = field(default=False, init=False)

    #: Max bytes of the text block emitted in :attr:`content_blocks`.
    MAX_TEXT_BYTES_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        self.content_blocks = self._build_content_blocks()
        self.is_finish = bool(
            isinstance(self.result, dict) and self.result.get("_finish")
        )

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        """Build Anthropic-shaped content blocks (text + optional images).

        Strips image byte payloads from the text block and emits them as
        separate base64 image blocks so the LLM receives the state images as
        multimodal content.
        """
        result = self.result
        if not isinstance(result, dict):
            return [
                {
                    "type": "text",
                    "text": _truncate_utf8(
                        str(result),
                        self.MAX_TEXT_BYTES_IN_RESULT,
                    ),
                }
            ]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_nav = result_for_text.pop("_image_nav_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        text = json.dumps(result_for_text, indent=2, default=str)
        text = _truncate_utf8(
            text,
            self.MAX_TEXT_BYTES_IN_RESULT,
            marker="\n[truncated]",
        )

        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": data,
                    },
                }
            )

        if image:
            _add_image_bytes(image)
        if image_cam:
            _add_image_bytes(image_cam)
        if image_nav:
            _add_image_bytes(image_nav)
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Robot-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying env Primitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release robot-side primitives / servers at the end of the run.
    """

    def __init__(
        self,
        *,
        dashboard_events: DashboardEventSink,
        state: Any = None,
        memory: "MemoryManager",
        recovery_goal: str | None = None,
    ) -> None:
        self._tools: dict[
            str,
            tuple[dict[str, Any], Callable[..., Any]],
        ] = {}
        self._dashboard_events = dashboard_events
        self._state = state
        self._memory = memory
        self._operation_lock = threading.Lock()
        self._active_operation: _ToolOperation | None = None
        self._recovery_goal = recovery_goal
        self._recovery_episode_id: str | None = None
        self._recovery_writer: JsonlWriter | None = None
        self._recovery_unavailable_logged = False
        self._register_common_tools()
        self._init_recovery_observer()

    def _init_recovery_observer(self) -> None:
        """Open this run's recovery ledger when a real goal is available."""
        if not isinstance(self._recovery_goal, str) or not self._recovery_goal.strip():
            return
        try:
            output_dir = get_output_dir().resolve()
            state_output_dir = self.state.output_dir.resolve()
            state_output_dir.relative_to(output_dir)
            self._recovery_episode_id = state_output_dir.relative_to(
                output_dir.parent
            ).as_posix()
            from rpent.recovery.persistence import JsonlWriter

            self._recovery_writer = JsonlWriter(
                state_output_dir / "recovery_events.jsonl"
            )
        except Exception as exc:
            logger.warning(
                "recovery observer unavailable during toolkit setup: %s", exc
            )

    def _observe_tool_failure(
        self,
        *,
        name: str,
        result: dict[str, Any],
        elapsed_s: float,
        record: StepRecord | None,
        failure_source: str,
        tool_error: str | None,
        state_capture_error: str | None,
    ) -> None:
        """Diagnose and persist one failure without taking recovery action."""
        if self._recovery_writer is None or self._recovery_episode_id is None:
            if not self._recovery_unavailable_logged:
                logger.warning(
                    "recovery event not recorded for %s: observer has no goal or writer",
                    name,
                )
                self._recovery_unavailable_logged = True
            return
        if record is None:
            logger.warning(
                "recovery event not recorded for %s: no environment step is available",
                name,
            )
            return

        from rpent.recovery.diagnose import DiagnosisSignals, FailureDiagnoser
        from rpent.recovery.events import ExecutionEvent, RecoveryDecision
        from rpent.recovery.libero_evidence import (
            libero_tool_evidence,
            map_libero_evidence,
        )
        from rpent.recovery.router import FailureRouter

        tool_result = record.result if isinstance(record.result, dict) else {}
        mapped = map_libero_evidence(
            {"state": record.state, "log": {"result": tool_result}}
        )
        signals = DiagnosisSignals(
            libero_predicate=record.terminated,
            tool_error=tool_error,
            end_effector_pose=mapped.get("end_effector_pose"),
            gripper_opening=mapped.get("gripper_opening"),
            libero_tool_evidence=libero_tool_evidence(mapped),
            transcript_text=(
                tool_result.get("transcript_text")
                if isinstance(tool_result.get("transcript_text"), str)
                else None
            ),
        )
        diagnosis = FailureDiagnoser().diagnose(signals)
        metadata: dict[str, Any] = {
            "elapsed_s": elapsed_s,
            "failure_source": failure_source,
        }
        if state_capture_error is not None:
            metadata["state_capture_error"] = state_capture_error
        base = ExecutionEvent(
            episode_id=self._recovery_episode_id,
            goal=self._recovery_goal,
            step=record.step_idx,
            state=record.state,
            tool_id=name,
            outcome="failed",
            metadata=metadata,
        )
        failure = diagnosis.to_failure_event(base)
        routed = FailureRouter().route(
            failure,
            available_tool_ids=self._tools,
        )
        decision = RecoveryDecision(
            level=routed.level,
            action=routed.action,
            reason=routed.reason,
            evidence=routed.evidence,
            source=routed.source,
            event_id=failure.event_id,
            episode_id=failure.episode_id,
            step=failure.step,
            cost={"wall_clock_s": elapsed_s},
            termination_reason=routed.termination_reason,
            attempts=routed.attempts,
        )
        self._recovery_writer.append(
            {
                "event": failure.to_manifest(),
                "decision": decision.to_manifest(),
            }
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict. Decorate read-only handlers with
                :func:`readonly`; all other handlers capture state.
        """
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        memory_bindings = self._memory.get_common_tool_bindings()
        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            binding = memory_bindings.get(name)
            if binding is None:
                binding = (spec, common.TOOL_HANDLERS[name])
            tool_spec, handler = binding
            self.add_tool(name, tool_spec, handler)

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    @property
    def memory(self) -> "MemoryManager":
        """Return the toolkit's memory manager."""
        return self._memory

    @property
    def state(self) -> EnvState:
        """Return the run's artifact and step store."""
        if self._state is None:
            raise RuntimeError("toolkit has no environment state")
        return self._state

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute([spec for spec, _ in self._tools.values()])

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        _, handler = entry

        with self._operation_lock:
            if self._active_operation is not None:
                return ToolResult(
                    name=name,
                    result={"error": "another tool operation is still active"},
                )
            operation = _ToolOperation()
            self._active_operation = operation

        try:
            started = time.perf_counter()
            handler_failed = False
            observer_failed = False
            failure_source: str | None = None
            tool_error: str | None = None
            state_capture_error: str | None = None
            record: StepRecord | None = None
            record_before = (
                self._state.latest_record() if self._state is not None else None
            )
            try:
                result = handler(**input_dict)
            except TypeError as e:
                result = {
                    "error": f"bad arguments for {name}: {e}",
                    "got": input_dict,
                }
                handler_failed = True
                observer_failed = True
                failure_source = "exception"
                tool_error = str(e)
            except ToolCancelled as e:
                result = {
                    "error": str(e),
                    "code": "tool_cancelled",
                    "interrupted": True,
                }
                handler_failed = True
                observer_failed = True
                failure_source = "exception"
                tool_error = str(e)
            except Exception as e:
                result = {"error": str(e), "traceback": traceback.format_exc()}
                handler_failed = True
                observer_failed = True
                failure_source = "exception"
                tool_error = str(e)

            result_dict = result if isinstance(result, dict) else {"value": result}
            if not handler_failed:
                from rpent.recovery.tool_result import classify_tool_result_failure

                failure_source, error_value = classify_tool_result_failure(result_dict)
                observer_failed = failure_source not in {None, "task_not_terminated"}
                if failure_source == "task_not_terminated":
                    logger.debug(
                        "%s reported task not terminated after a normal contact skill",
                        name,
                    )
                tool_error = str(error_value) if error_value is not None else None

            if not _is_readonly(handler):
                elapsed_s = round(time.perf_counter() - started, 2)
                command = {"action": name, **input_dict}
                try:
                    captured = self.get_env_state(
                        command=command,
                        result=result_dict,
                        elapsed_s=elapsed_s,
                    )
                except Exception as e:
                    state_capture_error = str(e)
                    captured = result_dict
                    captured["state_capture_error"] = state_capture_error
                    captured.setdefault(
                        "error", f"failed to capture state after {name}: {e}"
                    )
                    captured.setdefault("traceback", traceback.format_exc())
                    if not observer_failed:
                        observer_failed = True
                        failure_source = "state_capture"
                finally:
                    latest_record = (
                        self._state.latest_record() if self._state is not None else None
                    )
                    if latest_record is not record_before:
                        record = latest_record
                result = captured
                if handler_failed:
                    for key, value in result_dict.items():
                        result.setdefault(key, value)
                if record is not None:
                    self._publish_step(record)

            result_dict = result if isinstance(result, dict) else {"value": result}

            if observer_failed:
                elapsed_s = round(time.perf_counter() - started, 2)
                if failure_source is None:
                    logger.error(
                        "recovery event not recorded for failed tool %s: "
                        "failure source was not set",
                        name,
                    )
                elif _is_readonly(handler):
                    logger.warning(
                        "recovery event not recorded for readonly tool %s: "
                        "no environment step was produced",
                        name,
                    )
                else:
                    try:
                        self._observe_tool_failure(
                            name=name,
                            result=result_dict,
                            elapsed_s=elapsed_s,
                            record=record,
                            failure_source=failure_source,
                            tool_error=tool_error,
                            state_capture_error=state_capture_error,
                        )
                    except Exception as exc:
                        logger.warning(
                            "recovery observer failed for tool %s: %s", name, exc
                        )

            return ToolResult(name=name, result=result)
        finally:
            with self._operation_lock:
                self._active_operation = None
                operation.done_event.set()

    def _publish_step(self, record: StepRecord) -> None:
        """Publish one recorded environment step to the dashboard sink."""
        self._dashboard_events.emit(
            StepRecordEvent(
                record=record,
                env_state=self._state,
            )
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        """Capture and return the observation produced by a stateful tool."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Server lifecycle hooks (overridden by robot toolkits)
    # ------------------------------------------------------------------

    def cancel_active_and_wait(self) -> None:
        """Request cancellation and wait for the active tool to return."""
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        operation.done_event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise at an environment-defined safe cancellation boundary."""
        with self._operation_lock:
            operation = self._active_operation
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted")

    def close(self) -> None:
        """Release the robot-side primitives / servers at end of run.

        The base implementation now releases the recovery ledger; subclass
        overrides must call ``super().close()``.
        """
        writer = getattr(self, "_recovery_writer", None)
        if writer is None:
            return
        try:
            writer.close()
        except Exception as exc:
            logger.warning("failed to close recovery observer: %s", exc)
        finally:
            self._recovery_writer = None

    def solved(self) -> bool:
        """Whether the env has reported the task complete.

        Ground truth for the session loop: an agent may call ``finish`` with
        ``status="success"`` on a cell it did not actually finish, so the
        handoff decision reads the environment, not the agent.
        """
        raise NotImplementedError

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this robot, if supported."""
        return None
