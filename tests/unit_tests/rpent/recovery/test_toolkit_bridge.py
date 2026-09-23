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

import pytest

from rpent.memory import MemoryManager
from rpent.recovery import (
    MockToolSynthesizer,
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    ToolCall,
    ToolError,
    ToolGapAdapter,
    ToolGapEvent,
    ToolkitBackedRegistry,
    ToolSpec,
    ToolVerifier,
    VerificationReport,
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
        self.add_tool(
            "falsy_error",
            {
                "name": "falsy_error",
                "description": "Return a falsy error value",
                "input_schema": {"type": "object"},
            },
            self._falsy_error,
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

    @staticmethod
    @readonly
    def _falsy_error(error: Any) -> dict[str, Any]:
        return {"error": error, "done": True}

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


class _RegistryToolkit(_ChangingFailureToolkit):
    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return dict(result)


class _StrictParameterToolkit(Toolkit):
    def __init__(self, output_dir: Path) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.move_calls = 0
        init_output_dir(output_dir / "logs")
        super().__init__(
            dashboard_events=_RecordingEventSink(),
            state=EnvState(output_dir),
            memory=MemoryManager(output_dir / "memory"),
        )
        schemas = {
            "move": self._move,
            "place": self._place,
            "undeclared_pose": self._undeclared_pose,
        }
        for name, handler in schemas.items():
            self.add_tool(
                name,
                {
                    "name": name,
                    "description": f"Strict {name} handler",
                    "input_schema": {"type": "object"},
                },
                handler,
            )

    def _move(self, pose: list[int]) -> dict[str, bool]:
        self.calls.append(("move", {"pose": pose}))
        self.move_calls += 1
        if self.move_calls == 1:
            raise RuntimeError("bad pose")
        return {"moved": True}

    def _place(self, other: int) -> dict[str, bool]:
        self.calls.append(("place", {"other": other}))
        return {"placed": True}

    def _undeclared_pose(self, other: int) -> dict[str, bool]:
        self.calls.append(("undeclared_pose", {"other": other}))
        raise RuntimeError("bad pose")

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return {
            **result,
            "end_effector_pose": {
                "reachable": False,
                "target_pose": [1, 0, 0],
            },
        }

    def solved(self) -> bool:
        return False


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


def test_bridge_treats_falsy_error_values_as_success(tmp_path: Path) -> None:
    registry = ToolkitBackedRegistry(_ChangingFailureToolkit(tmp_path))

    for error in (None, ""):
        result = registry.invoke(ToolCall("falsy_error", {"error": error}))
        assert result.success is True
        assert result.error is None


def test_bridge_registers_and_invokes_recovery_executor(tmp_path: Path) -> None:
    registry = ToolkitBackedRegistry(_RegistryToolkit(tmp_path))
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    spec = ToolSpec(
        "generated.move",
        "Generated move",
        "Move through a generated executor",
        input_schema={"type": "object"},
    )

    def executor(arguments: Any, context: Any) -> dict[str, Any]:
        calls.append((dict(arguments), dict(context)))
        return {"received": dict(arguments)}

    registry.register(spec, executor)
    result = registry.invoke(ToolCall("generated.move", {"distance": 2}), {"live": 1})

    assert registry.has("generated.move")
    assert registry.get("generated.move") == spec
    assert "generated.move" in {item.tool_id for item in registry.list_specs()}
    assert result.success is True
    assert result.output == {"received": {"distance": 2}}
    assert calls == [({"distance": 2}, {})]


def test_bridge_distinguishes_register_and_replace(tmp_path: Path) -> None:
    registry = ToolkitBackedRegistry(_RegistryToolkit(tmp_path))
    spec = ToolSpec("generated.echo", "Echo", "Echo a version")
    registry.register(spec, lambda arguments, context: {"version": 1})

    with pytest.raises(ToolError, match="already registered"):
        registry.register(spec, lambda arguments, context: {"version": 2})
    with pytest.raises(ToolError, match="cannot replace unknown"):
        registry.replace(
            ToolSpec("generated.unknown", "Unknown", "Unknown tool"),
            lambda arguments, context: {},
        )

    registry.replace(spec, lambda arguments, context: {"version": 2})
    assert registry.invoke(ToolCall(spec.tool_id)).output == {"version": 2}


def test_bridge_reuses_verified_registration_gate(tmp_path: Path) -> None:
    registry = ToolkitBackedRegistry(_RegistryToolkit(tmp_path))
    spec = ToolSpec("generated.safe", "Safe", "Verified tool")

    def executor(arguments: Any, context: Any) -> dict[str, bool]:
        return {"safe": True}

    with pytest.raises(ToolError, match="does not belong"):
        registry.register_verified(
            spec,
            executor,
            VerificationReport("another-tool", True),
        )
    with pytest.raises(ToolError, match="did not pass"):
        registry.register_verified(
            spec, executor, VerificationReport(spec.tool_id, False)
        )

    registry.register_verified(spec, executor, VerificationReport(spec.tool_id, True))
    assert registry.has(spec.tool_id)


def test_bridge_manifest_round_trip(tmp_path: Path) -> None:
    registry = ToolkitBackedRegistry(_RegistryToolkit(tmp_path))
    spec = ToolSpec("generated.saved", "Saved", "Persisted descriptor")
    registry.register(spec, lambda arguments, context: {})
    path = tmp_path / "tools.json"

    registry.save_manifest(path)

    assert registry.manifest()["schema"] == "agentic-em/tool-manifest/v1"
    loaded = registry.load_specs(path)
    assert spec.tool_id in {item.tool_id for item in loaded}


def test_tool_gap_coordinator_registers_into_bridge(tmp_path: Path) -> None:
    registry = ToolkitBackedRegistry(_RegistryToolkit(tmp_path))
    adapter = ToolGapAdapter(registry)
    gap = ToolGapEvent(
        episode_id="bridge-l3",
        goal="restage",
        step=0,
        failure_family="tool_gap",
        diagnosis={},
        required_capability="restage",
        missing_capability="restage",
    )

    synthesis = adapter.synthesize_tool_gap(
        gap,
        MockToolSynthesizer(),
        [({"object": "mug"}, {})],
        postcondition=lambda output, context: output.get("solved") is True,
        verifier=ToolVerifier(),
    )
    invoked = registry.invoke(
        ToolCall("capx.restage", {"object": "mug"}), {"ignored": True}
    )

    assert synthesis.registered is True
    assert registry.has("capx.restage")
    assert invoked.success is True
    assert invoked.output == {"capability": "restage", "solved": True}


def test_bridge_derives_description_from_name_when_schema_description_is_blank(
    tmp_path: Path,
) -> None:
    toolkit = _RegistryToolkit(tmp_path)
    for name, description in (
        ("description_missing", None),
        ("description_blank", "   "),
    ):
        schema = {"name": name, "input_schema": {"type": "object"}}
        if description is not None:
            schema["description"] = description
        toolkit.add_tool(name, schema, lambda: {})

    registry = ToolkitBackedRegistry(toolkit)

    assert registry.get("description_missing").description == "description_missing"
    assert registry.get("description_blank").description == "description_blank"


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


def test_l1_adaptation_does_not_leak_into_later_strict_step(tmp_path: Path) -> None:
    toolkit = _StrictParameterToolkit(tmp_path)
    skill = SkillPlaybook(
        skill_id="strict-cross-step",
        name="Strict cross-step",
        goal="move then place",
        trigger_labels=frozenset({"bad_pose"}),
        diagnosis="correct the move pose",
        steps=(
            SkillStep("move", "move", "move", {"pose": [0, 0, 0]}),
            SkillStep("place", "place", "place", {"other": 1}),
        ),
        verification_checks=("placed",),
    )

    result = SkillRuntime(ToolkitBackedRegistry(toolkit), recovery_loop=True).execute(
        skill, episode_id="strict-cross-step"
    )

    assert result.success is True
    assert toolkit.calls == [
        ("move", {"pose": [0, 0, 0]}),
        ("move", {"pose": [1, 0, 0]}),
        ("place", {"other": 1}),
    ]


def test_l1_does_not_inject_undeclared_argument_into_strict_step(
    tmp_path: Path,
) -> None:
    toolkit = _StrictParameterToolkit(tmp_path)
    skill = SkillPlaybook(
        skill_id="strict-current-step",
        name="Strict current-step",
        goal="reject undeclared pose",
        trigger_labels=frozenset({"bad_pose"}),
        diagnosis="pose evidence does not declare an argument",
        steps=(
            SkillStep(
                "move",
                "move",
                "undeclared_pose",
                {"other": 7},
            ),
        ),
        verification_checks=("moved",),
    )

    result = SkillRuntime(ToolkitBackedRegistry(toolkit), recovery_loop=True).execute(
        skill, episode_id="strict-current-step"
    )

    assert result.success is False
    assert toolkit.calls == [("undeclared_pose", {"other": 7})]
    assert result.error == "bad pose"
    assert result.decision.termination_reason == "L1 无可用自适应证据"
