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

from pathlib import Path
from typing import Any

from rpent.memory import MemoryManager
from rpent.recovery import (
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    ToolCall,
    ToolkitBackedRegistry,
)
from rpent.recovery.adapt import ParameterAdaptation, ParameterAdapter
from rpent.recovery.diagnose import DiagnosisResult
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import init_output_dir


class _RecordingEventSink:
    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        pass


class _ChangingFailureToolkit(Toolkit):
    def __init__(self, output_dir: Path) -> None:
        self.capture_count = 0
        init_output_dir(output_dir / "logs")
        super().__init__(
            dashboard_events=_RecordingEventSink(),
            state=EnvState(output_dir),
            memory=MemoryManager(output_dir / "memory"),
        )
        self.add_tool(
            "move",
            {
                "name": "move",
                "description": "Always fail after attempting a pose",
                "input_schema": {"type": "object"},
            },
            self._fail,
        )
        self.add_tool(
            "scalar",
            {
                "name": "scalar",
                "description": "Return a non-mapping value",
                "input_schema": {"type": "object"},
            },
            self._scalar,
        )
        self.add_tool(
            "finish_with_image",
            {
                "name": "finish_with_image",
                "description": "Return a finish marker and image",
                "input_schema": {"type": "object"},
            },
            self._finish_with_image,
        )
        self.add_tool(
            "reported_failure",
            {
                "name": "reported_failure",
                "description": "Return an explicit task failure",
                "input_schema": {"type": "object"},
            },
            self._reported_failure,
        )
        self.add_tool(
            "reported_success",
            {
                "name": "reported_success",
                "description": "Return an explicit task success",
                "input_schema": {"type": "object"},
            },
            self._reported_success,
        )

    def _fail(self, pose: list[int]) -> dict[str, Any]:
        raise RuntimeError("fixed failure")

    @staticmethod
    @readonly
    def _scalar() -> str:
        return "value"

    @staticmethod
    @readonly
    def _finish_with_image() -> dict[str, Any]:
        return {"_finish": True, "_image_wrist_bytes": b"wrist", "done": True}

    @staticmethod
    @readonly
    def _reported_failure() -> dict[str, Any]:
        return {"success": False, "diagnostics": {"reason": "missed"}}

    @staticmethod
    @readonly
    def _reported_success() -> dict[str, Any]:
        return {"success": True}

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        self.capture_count += 1
        return {
            "end_effector_pose": {
                "reachable": False,
                "target_pose": [self.capture_count, 0, 0],
            },
            "_image_bytes": b"main",
            "_image_cam_bytes": b"camera",
            "_image_nav_bytes": b"navigation",
        }

    def solved(self) -> bool:
        return False


class _RecordingAdapter(ParameterAdapter):
    def __init__(self) -> None:
        self.adaptations: list[ParameterAdaptation] = []

    def adapt(
        self, diagnosis: DiagnosisResult, arguments: dict[str, Any]
    ) -> ParameterAdaptation:
        adaptation = super().adapt(diagnosis, arguments)
        self.adaptations.append(adaptation)
        return adaptation


def test_bridge_preserves_failure_output_and_normalizes_toolkit_metadata(
    tmp_path: Path,
) -> None:
    registry = ToolkitBackedRegistry(_ChangingFailureToolkit(tmp_path))

    assert registry.has("move")
    assert "scalar" in {spec.tool_id for spec in registry.list_specs()}

    failure = registry.invoke(ToolCall("move", {"pose": [0, 0, 0]}), {})
    assert failure.success is False
    assert failure.error == "fixed failure"
    assert failure.output["error"] == "fixed failure"
    assert failure.output["end_effector_pose"]["target_pose"] == [1, 0, 0]
    assert failure.output["_images"] == [
        "_image_bytes",
        "_image_cam_bytes",
        "_image_nav_bytes",
    ]
    assert all(
        key not in failure.output
        for key in (
            "_image_bytes",
            "_image_cam_bytes",
            "_image_nav_bytes",
            "_image_wrist_bytes",
        )
    )
    assert failure.metadata["wall_clock_s"] >= 0
    assert failure.metadata["is_finish"] is False
    assert failure.metadata["call_id"] is None

    scalar = registry.invoke(ToolCall("scalar"))
    assert scalar.success is True
    assert scalar.output == {"value": "value"}

    finish = registry.invoke(ToolCall("finish_with_image"))
    assert finish.success is True
    assert finish.output == {
        "_finish": True,
        "done": True,
        "_images": ["_image_wrist_bytes"],
    }
    assert finish.metadata["is_finish"] is True


def test_bridge_honors_explicit_success_without_reclassifying_other_results(
    tmp_path: Path,
) -> None:
    toolkit = _ChangingFailureToolkit(tmp_path)
    direct = toolkit.execute_tool("reported_failure", {})
    assert direct.result["success"] is False

    registry = ToolkitBackedRegistry(toolkit)

    reported_failure = registry.invoke(ToolCall("reported_failure"))
    assert reported_failure.success is False
    assert reported_failure.error == "tool result reported success=False"

    reported_success = registry.invoke(ToolCall("reported_success"))
    assert reported_success.success is True
    assert reported_success.error is None

    ordinary = registry.invoke(ToolCall("finish_with_image"))
    assert ordinary.success is True


def test_changing_failure_state_produces_two_distinct_l1_adaptations(
    tmp_path: Path,
) -> None:
    registry = ToolkitBackedRegistry(_ChangingFailureToolkit(tmp_path))
    adapter = _RecordingAdapter()
    skill = SkillPlaybook(
        skill_id="changing-pose",
        name="Changing pose",
        goal="move",
        trigger_labels=frozenset({"bad_pose"}),
        diagnosis="pose remains unreachable",
        steps=(SkillStep("move", "move to pose", "move", {"pose": [0, 0, 0]}),),
        verification_checks=("moved",),
    )

    result = SkillRuntime(
        registry,
        recovery_loop=True,
        parameter_adapter=adapter,
    ).execute(skill, episode_id="bridge-l1")

    assert len(adapter.adaptations) >= 2
    assert adapter.adaptations[0].delta == {"pose": [1, 0, 0]}
    assert adapter.adaptations[1].delta == {"pose": [2, 0, 0]}
    assert adapter.adaptations[0].delta != adapter.adaptations[1].delta
    assert result.decision is not None
    assert (
        result.decision.termination_reason
        != "L1 adaptation produced no argument change"
    )
