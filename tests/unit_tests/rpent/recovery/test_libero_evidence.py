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

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.memory import MemoryManager
from rpent.recovery import (
    DiagnosisSignals,
    FailureDiagnoser,
    RecoveryLevel,
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    ToolkitBackedRegistry,
    map_libero_evidence,
)
from rpent.recovery.libero_evidence import libero_tool_evidence
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import init_output_dir


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


def test_libero_move_pose_reports_orientation_without_inventing_error() -> None:
    mapped = map_libero_evidence(
        {
            "name": "move_pose",
            "success": False,
            "final_dist_m": 0.01,
            "final_pitch": 0.45,
            "diagnostics": {"tol": 0.012, "ori_tol": 0.05},
        }
    )

    assert mapped["libero_position"]["value"]["reached"] is True
    assert mapped["libero_orientation"] == {
        "source": (
            "LIBERO move_pose result: success=False while final_dist_m < "
            "diagnostics.tol; "
            "final_pitch and diagnostics.ori_tol"
        ),
        "value": {"reached": False, "final_pitch": 0.45, "ori_tol": 0.05},
    }
    assert "error" not in mapped["libero_orientation"]["value"]


def test_libero_tool_results_without_returned_thresholds_make_no_judgment() -> None:
    mapped = map_libero_evidence(
        {
            "name": "move_to",
            "success": False,
            "target_xyz": [0.1, 0.0, 0.0],
            "final_eef_pos": [0.0, 0.0, 0.0],
            "final_dist_m": 0.1,
            "diagnostics": {},
        }
    )
    diagnosis = FailureDiagnoser().diagnose(
        DiagnosisSignals(
            end_effector_pose=mapped.get("end_effector_pose"),
            libero_tool_evidence=libero_tool_evidence(mapped),
        )
    )

    assert "libero_position" not in mapped
    assert "reachable" not in mapped["end_effector_pose"]
    assert diagnosis.failure_family == "unknown"


def test_libero_tool_evidence_rejects_existing_evidence_key() -> None:
    with pytest.raises(ValueError, match="conflicts with 'libero_predicate'"):
        FailureDiagnoser().diagnose(
            DiagnosisSignals(
                libero_tool_evidence={
                    "libero_predicate": {"source": "wrong source", "value": False}
                }
            )
        )


def test_libero_pick_failure_subcriteria_use_tool_thresholds_in_order() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "libero90_smoke_steps.json").read_text()
    )
    sample = next(
        step["result"]
        for step in fixture["steps"]
        if step["task"] == 2 and step["step_idx"] == 11
    )
    no_descent = map_libero_evidence(sample)
    diagnosis = FailureDiagnoser().diagnose(
        DiagnosisSignals(libero_tool_evidence=libero_tool_evidence(no_descent))
    )
    assert diagnosis.evidence["rule_id"]["value"] == "libero_pick_no_descent"

    no_close_sample = copy.deepcopy(sample)
    no_close_sample["diagnostics"]["descent_m"] = 0.1
    no_close = map_libero_evidence(no_close_sample)
    diagnosis = FailureDiagnoser().diagnose(
        DiagnosisSignals(libero_tool_evidence=libero_tool_evidence(no_close))
    )
    assert diagnosis.failure_family == "grasp_contact"
    assert diagnosis.evidence["rule_id"]["value"] == "libero_pick_no_close"

    no_lift_sample = copy.deepcopy(no_close_sample)
    no_lift_sample["min_gripper_opening"] = 0.03
    no_lift = map_libero_evidence(no_lift_sample)
    diagnosis = FailureDiagnoser().diagnose(
        DiagnosisSignals(libero_tool_evidence=libero_tool_evidence(no_lift))
    )
    assert diagnosis.failure_family == "grasp_contact"
    assert diagnosis.evidence["rule_id"]["value"] == "libero_pick_no_lift"


def test_pi0_doubled_mode_matches_libero_tool_contract() -> None:
    source = Path("robots/libero/tools.py").read_text()
    from rpent.recovery.tool_result import CONTACT_SKILL_SUCCESS_BY_TERMINATION

    assert CONTACT_SKILL_SUCCESS_BY_TERMINATION in source


class _FixtureToolkit(Toolkit):
    def __init__(self, output_dir: Path, steps: list[dict[str, Any]]) -> None:
        self.pending_steps = iter(steps)
        self.current_step: dict[str, Any] | None = None
        super().__init__(
            dashboard_events=SimpleNamespace(enabled=False, emit=lambda event: None),
            state=EnvState(output_dir),
            memory=MemoryManager(output_dir / "memory"),
            recovery_goal="LIBERO smoke fixture",
        )
        self.add_tool("move_to", {"name": "move_to"}, self._move_to)
        self.add_tool("move_pose", {"name": "move_pose"}, self._move_pose)
        self.add_tool("pi0_pick", {"name": "pi0_pick"}, self._pi0_pick)
        self.add_tool("pi0_doubled", {"name": "pi0_doubled"}, self._pi0_doubled)
        self.add_tool("release", {"name": "release"}, self._release)
        self.add_tool("set_gripper", {"name": "set_gripper"}, self._set_gripper)

    def _result(self) -> dict[str, Any]:
        self.current_step = next(self.pending_steps)
        return copy.deepcopy(self.current_step["result"])

    def _move_to(self, xyz, *, max_steps, gripper, step_clip, tol):
        return self._result()

    def _move_pose(
        self, xyz, *, target_pitch, target_yaw, gripper, step_clip, tol, max_steps
    ):
        return self._result()

    def _pi0_pick(self, prompt, *, max_chunks, lift_thresh, gripper_closed_thresh):
        return self._result()

    def _pi0_doubled(self, prompt, *, max_chunks):
        return self._result()

    def _release(self, *, max_steps):
        return self._result()

    def _set_gripper(self, *, gripper, steps):
        return self._result()

    def get_env_state(self, *, command, result, elapsed_s):
        assert self.current_step is not None
        expected = self.current_step
        assert command == expected["command"]
        while self.state.latest_record() is None or (
            self.state.latest_record().step_idx < expected["step_idx"] - 1
        ):
            with self.state.record_step(state={"fixture_gap": True}):
                pass
        with self.state.record_step(
            state=expected["state"],
            terminated=expected["terminated"],
            truncated=expected["truncated"],
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            pass
        return {"state": expected["state"], "terminated": expected["terminated"]}

    def solved(self) -> bool:
        return False


EXPECTED_SMOKE_OUTCOMES = [
    (0, 1, "move_to", None),
    (0, 2, "pi0_doubled", None),
    (0, 3, "set_gripper", None),
    (0, 4, "move_to", None),
    (0, 5, "move_to", None),
    (0, 6, "move_to", None),
    (0, 7, "move_to", None),
    (0, 8, "move_to", None),
    (0, 9, "move_to", None),
    (0, 10, "move_to", None),
    (0, 11, "pi0_doubled", None),
    (0, 12, "set_gripper", None),
    (0, 13, "set_gripper", None),
    (0, 15, "pi0_doubled", None),
    (1, 1, "move_to", None),
    (1, 2, "set_gripper", None),
    (
        1,
        3,
        "move_to",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
    (1, 4, "move_to", None),
    (1, 5, "pi0_pick", None),
    (1, 6, "set_gripper", None),
    (1, 7, "move_to", None),
    (1, 8, "move_to", None),
    (1, 9, "move_to", None),
    (1, 10, "move_to", None),
    (1, 11, "release", None),
    (1, 12, "move_to", None),
    (
        1,
        13,
        "move_to",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
    (1, 14, "pi0_doubled", None),
    (
        2,
        1,
        "move_pose",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
    (2, 2, "pi0_pick", None),
    (2, 3, "set_gripper", None),
    (2, 4, "move_to", None),
    (2, 5, "move_pose", None),
    (2, 6, "move_pose", None),
    (
        2,
        7,
        "move_pose",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
    (2, 8, "release", None),
    (2, 9, "move_to", None),
    (
        2,
        10,
        "move_pose",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
    (
        2,
        11,
        "pi0_pick",
        ("parameter_or_pose", "libero_pick_no_descent", "L1", "adapt_parameters"),
    ),
    (
        2,
        13,
        "move_pose",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
    (2, 14, "pi0_pick", None),
    (2, 15, "set_gripper", None),
    (2, 16, "move_to", None),
    (2, 17, "move_pose", None),
    (
        2,
        18,
        "move_pose",
        ("parameter_or_pose", "libero_position_not_reached", "L1", "adapt_parameters"),
    ),
]


def test_libero90_fixture_observer_and_router_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "libero90_smoke_steps.json").read_text()
    )
    actual = []
    for task in (0, 1, 2):
        task_steps = [step for step in fixture["steps"] if step["task"] == task]
        output_dir = tmp_path / f"task_{task}"
        monkeypatch.setattr("rpent.tools.toolkit.get_output_dir", lambda: tmp_path)
        toolkit = _FixtureToolkit(output_dir, task_steps)
        event_count = 0
        for step in task_steps:
            action = step["command"]["action"]
            arguments = {k: v for k, v in step["command"].items() if k != "action"}
            assert set(arguments) == set(
                inspect.signature(toolkit._tools[action][1]).parameters
            )
            toolkit.execute_tool(action, arguments)
            records = [
                json.loads(line)
                for line in (output_dir / "recovery_events.jsonl")
                .read_text()
                .splitlines()
            ]
            outcome = None
            if len(records) > event_count:
                record = records[-1]
                event = record["event"]
                decision = record["decision"]
                outcome = (
                    event["failure_family"],
                    event["diagnosis"]["rule_id"]["value"],
                    decision["level"],
                    decision["action"],
                )
                assert event["step"] == step["step_idx"]
                assert event["tool_id"] == action
                event_count += 1
            actual.append((task, step["step_idx"], action, outcome))
        toolkit.close()

    assert actual == EXPECTED_SMOKE_OUTCOMES


FAILURE_EXPECTATIONS = [
    (
        1,
        3,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (
        1,
        13,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (
        2,
        1,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (
        2,
        7,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (
        2,
        10,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (2, 11, "parameter_or_pose", "libero_pick_no_descent", "L1", "adapt_parameters"),
    (
        2,
        13,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (
        2,
        18,
        "parameter_or_pose",
        "libero_position_not_reached",
        "L1",
        "adapt_parameters",
    ),
    (
        2,
        "orientation",
        "parameter_or_pose",
        "libero_orientation_not_reached",
        "L1",
        "adapt_parameters",
    ),
]


class _RuntimeFixtureToolkit(_FixtureToolkit):
    def get_env_state(self, *, command, result, elapsed_s):
        captured = super().get_env_state(
            command=command, result=result, elapsed_s=elapsed_s
        )
        return {**captured, "log": {"result": result}}


@pytest.mark.parametrize("expected", FAILURE_EXPECTATIONS)
def test_fixture_failure_observer_and_runtime_match_literal_expectation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected: tuple
) -> None:
    task, index, family, rule, level, action = expected
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "libero90_smoke_steps.json").read_text()
    )
    step = copy.deepcopy(
        next(
            item
            for item in fixture["steps"]
            if item["task"] == task
            and item["step_idx"] == (1 if index == "orientation" else index)
        )
    )
    if index == "orientation":
        step["result"]["final_dist_m"] = step["result"]["diagnostics"]["tol"] / 2
        assert step["result"]["success"] is False

    init_output_dir(tmp_path / "logs")
    monkeypatch.setattr("rpent.tools.toolkit.get_output_dir", lambda: tmp_path)
    command = step["command"]
    arguments = {key: value for key, value in command.items() if key != "action"}
    expected_outcome = (family, rule, level, action)
    for path in ("observer", "runtime"):
        toolkit = _RuntimeFixtureToolkit(tmp_path / f"{path}_{task}_{index}", [step])
        handler = toolkit._tools[command["action"]][1]
        assert set(arguments) == set(inspect.signature(handler).parameters)
        if path == "observer":
            toolkit.execute_tool(command["action"], arguments)
            record = json.loads(
                (tmp_path / f"{path}_{task}_{index}" / "recovery_events.jsonl")
                .read_text()
                .splitlines()[-1]
            )
            event, decision = record["event"], record["decision"]
            actual = (
                event["failure_family"],
                event["diagnosis"]["rule_id"]["value"],
                decision["level"],
                decision["action"],
            )
        else:
            skill = SkillPlaybook(
                skill_id="fixture-failure",
                name="Fixture failure",
                goal="exercise diagnosis",
                trigger_labels=frozenset({"fixture"}),
                diagnosis="fixture failure",
                steps=(
                    SkillStep("failing-step", "execute", command["action"], arguments),
                ),
                verification_checks=("success",),
            )
            result = SkillRuntime(ToolkitBackedRegistry(toolkit)).execute(
                skill, episode_id="fixture-failure"
            )
            assert result.decision is not None
            event, decision = result.events[-1], result.decision
            actual = (
                event.failure_family,
                event.diagnosis["rule_id"]["value"],
                decision.level.value,
                decision.action.value,
            )
        assert actual == expected_outcome, (path, task, index, actual)
        if index == "orientation":
            orientation = (
                event["diagnosis"]["libero_orientation"]
                if path == "observer"
                else event.diagnosis["libero_orientation"]
            )
            assert set(orientation["value"]) == {"reached", "final_pitch", "ori_tol"}
            assert "error" not in orientation["value"]
        toolkit.close()


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
