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

from types import SimpleNamespace
from typing import Any

from rpent.recovery import (
    RecoveryLevel,
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    ToolkitBackedRegistry,
    map_libero_evidence,
)


def test_libero_evidence_maps_pose_and_gripper_values() -> None:
    result = map_libero_evidence(
        {
            "success": False,
            "target_xyz": [0.2, 0.1, 0.0],
            "final_eef_pos": [0.1, 0.0, 0.05],
            "state": {
                "robot0_eef_pos": [0.1, 0.0, 0.05],
                "robot0_gripper_qpos": [0.03, -0.01],
            },
        }
    )
    assert result == {
        "end_effector_pose": {
            "reachable": False,
            "pose_error": [0.1, 0.1, -0.05],
            "current_pose": [0.1, 0.0, 0.05],
        },
        "gripper_opening": 0.04,
    }


def test_libero_evidence_uses_only_two_finger_joints() -> None:
    result = map_libero_evidence(
        {
            "state": {
                "robot0_gripper_qpos": [0.04, -0.038, 0.5, -0.5],
            }
        }
    )

    assert result["gripper_opening"] == 0.078
    assert "gripper_opening" not in map_libero_evidence(
        {"state": {"robot0_gripper_qpos": [0.04]}}
    )


class _RecordingRouter:
    def __init__(self) -> None:
        self.levels: list[RecoveryLevel] = []

    def route(self, event: Any, *, available_tool_ids: Any = ()) -> Any:
        from rpent.recovery import FailureRouter

        decision = FailureRouter().route(event, available_tool_ids=available_tool_ids)
        self.levels.append(decision.level)
        return decision


class _LiberoToolkit:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "move_to",
                "description": "move",
                "input_schema": {"type": "object"},
            }
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append(dict(arguments))
        final_x = 0.0 if len(self.calls) == 1 else 0.01
        return SimpleNamespace(
            result={
                "state": {
                    "robot0_eef_pos": [final_x, 0.0, 0.0],
                    "robot0_gripper_qpos": [0.04, -0.02],
                },
                "log": {
                    "result": {
                        "success": False,
                        "target_xyz": [0.1, 0.0, 0.0],
                        "final_eef_pos": [final_x, 0.0, 0.0],
                    }
                },
            },
            is_finish=False,
            call_id=None,
        )


def test_libero_failure_reaches_l1_and_applies_measured_residual() -> None:
    toolkit = _LiberoToolkit()
    registry = ToolkitBackedRegistry(toolkit)
    router = _RecordingRouter()
    skill = SkillPlaybook(
        skill_id="libero-move",
        name="LIBERO move",
        goal="move",
        trigger_labels=frozenset({"pose_failure"}),
        diagnosis="servo missed target",
        steps=(SkillStep("move", "move", "move_to", {"pose": [0.0, 0.0, 0.0]}),),
        verification_checks=("target reached",),
    )

    result = SkillRuntime(registry, router=router, recovery_loop=True).execute(
        skill, episode_id="libero-l1"
    )

    assert router.levels[:2] == [RecoveryLevel.L1, RecoveryLevel.L1]
    assert result.tool_results[0].success is False
    assert result.tool_results[0].error == "tool result reported success=False"
    assert result.tool_results[0].output["log"]["result"]["success"] is False
    assert toolkit.calls[0]["pose"] == [0.0, 0.0, 0.0]
    assert toolkit.calls[1]["pose"] == [0.1, 0.0, 0.0]
    assert toolkit.calls[2]["pose"] == [0.19, 0.0, 0.0]
    assert toolkit.calls[1]["pose"] != toolkit.calls[2]["pose"]
    assert result.events[1].parameter_issue is True
