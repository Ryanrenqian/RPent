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
    extract_vla_objects,
    pair_recovery_episodes,
    same_vla_subgoal,
    write_recovery_episodes,
)
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit

_FIXTURE = Path(__file__).parent / "fixtures" / "libero90_smoke_steps.json"
_OBJECT_NAMES = [
    "akita_black_bowl_1",
    "butter_1",
    "butter_2",
    "chocolate_pudding_1",
]
_TASK_LANGUAGES = {
    0: "close the top drawer of the cabinet",
    1: "close the top drawer of the cabinet and put the black bowl on top of it",
    2: "put the black bowl in the top drawer of the cabinet",
}
_RESET_STEPS = {0: 14, 2: 12}

EXPECTED_REAL_EPISODES = {
    0: [
        {
            "level": "cross_attempt",
            "success": (15, "pi0_doubled"),
            "failures": [
                (2, "pi0_doubled", None, None, "budget_exhausted_contact"),
                (11, "pi0_doubled", None, None, "budget_exhausted_contact"),
            ],
            "delta": list(range(3, 15)),
        }
    ],
    1: [],
    2: [
        {
            "level": "cross_attempt",
            "success": (14, "pi0_pick"),
            "failures": [
                (
                    11,
                    "pi0_pick",
                    "parameter_or_pose",
                    "libero_pick_no_descent",
                    "result_success_false",
                )
            ],
            "delta": [12, 13],
        }
    ],
}


class _ObserverToolkit(Toolkit):
    def __init__(self, output_dir: Path, steps: list[dict[str, Any]]) -> None:
        self.pending_steps = iter(steps)
        self.current_step: dict[str, Any] | None = None
        super().__init__(
            dashboard_events=SimpleNamespace(enabled=False, emit=lambda event: None),
            state=EnvState(output_dir),
            memory=MemoryManager(output_dir.parent.parent / "memory"),
            recovery_goal="LIBERO recovery episode fixture",
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


def _fixture_steps(task: int) -> list[dict[str, Any]]:
    fixture = json.loads(_FIXTURE.read_text())
    return [copy.deepcopy(step) for step in fixture["steps"] if step["task"] == task]


def _build_cell(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: int,
    *,
    reset: bool = True,
    final_predicate: bool = True,
    mutate: Any = None,
) -> Path:
    cell = root / f"task_{task}"
    session = cell / "sessions" / "session_001"
    steps = _fixture_steps(task)
    if not final_predicate:
        for step in steps:
            step["terminated"] = False
            step["result"]["terminated"] = False
    if mutate is not None:
        mutate(steps)
    monkeypatch.setattr("rpent.tools.toolkit.get_output_dir", lambda: cell)
    toolkit = _ObserverToolkit(session, steps)
    with toolkit.state.record_step(
        state={"object_names": _OBJECT_NAMES},
        extras={"task_language": _TASK_LANGUAGES[task]},
    ):
        pass
    reset_step = _RESET_STEPS.get(task) if reset else None
    for step in steps:
        if reset_step is not None and step["step_idx"] > reset_step:
            latest = toolkit.state.latest_record()
            if latest is not None and latest.step_idx == reset_step - 1:
                with toolkit.state.record_step(
                    state=copy.deepcopy(latest.state),
                    command={"action": "reset", "reason": "retry approach"},
                ):
                    pass
        command = step["command"]
        action = command["action"]
        arguments = {key: value for key, value in command.items() if key != "action"}
        handler = toolkit._tools[action][1]
        assert set(arguments) == set(inspect.signature(handler).parameters)
        toolkit.execute_tool(action, arguments)
    toolkit.close()
    return cell


def _summary(episodes) -> list[dict[str, Any]]:
    return [
        {
            "level": episode.level,
            "success": (episode.success_step, episode.success_tool_id),
            "failures": [
                (
                    failure.step,
                    failure.tool_id,
                    failure.failure_family,
                    failure.rule_id,
                    failure.failure_source,
                )
                for failure in episode.failures
            ],
            "delta": list(episode.delta_steps),
        }
        for episode in episodes
    ]


def test_libero_real_layout_has_exact_expected_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = {}
    for task in (0, 1, 2):
        cell = _build_cell(tmp_path, monkeypatch, task)
        actual[task] = _summary(pair_recovery_episodes(cell))
        if task == 0:
            assert (
                cell / "sessions/session_001/recovery_events.jsonl"
            ).read_text() == ""

    assert actual == EXPECTED_REAL_EPISODES
    assert sum(len(episodes) for episodes in actual.values()) == 2


def test_cross_attempt_becomes_intra_when_reset_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _build_cell(tmp_path, monkeypatch, 2, reset=False)

    assert _summary(pair_recovery_episodes(cell)) == [
        {**EXPECTED_REAL_EPISODES[2][0], "level": "intra_attempt"}
    ]
    states = json.loads((cell / "sessions/session_001/states.json").read_text())[
        "steps"
    ]
    assert [step["step_idx"] for step in states if 11 < step["step_idx"] < 14] == [
        12,
        13,
    ]
    assert all(step.get("command", {}).get("action") != "reset" for step in states)


def test_false_final_predicate_removes_cross_but_not_intra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cross_cell = _build_cell(tmp_path / "cross", monkeypatch, 2, final_predicate=False)
    intra_cell = _build_cell(
        tmp_path / "intra", monkeypatch, 2, reset=False, final_predicate=False
    )

    assert pair_recovery_episodes(cross_cell) == []
    assert _summary(pair_recovery_episodes(intra_cell))[0]["level"] == "intra_attempt"


@pytest.mark.parametrize(("offset", "expected_count"), [(0.06, 1), (0.060001, 0)])
def test_move_target_distance_uses_exact_five_times_tolerance_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offset: float,
    expected_count: int,
) -> None:
    def move_success(steps: list[dict[str, Any]]) -> None:
        failure = next(step for step in steps if step["step_idx"] == 3)
        success = next(step for step in steps if step["step_idx"] == 4)
        success["result"]["target_xyz"] = [
            failure["result"]["target_xyz"][0] + offset,
            *failure["result"]["target_xyz"][1:],
        ]

    cell = _build_cell(tmp_path, monkeypatch, 1, mutate=move_success)

    episodes = pair_recovery_episodes(cell)
    assert len(episodes) == expected_count
    if episodes:
        assert episodes[0].level == "intra_attempt"
        assert episodes[0].success_step == 4
        assert [failure.step for failure in episodes[0].failures] == [3]


def test_vla_object_matching_normalizes_drawer_and_distinguishes_surface() -> None:
    names = _OBJECT_NAMES
    language = "close the top drawer of the cabinet"
    close = {
        "command": {
            "action": "pi0_doubled",
            "prompt": "close the top drawer of the cabinet",
        },
        "state": {"object_names": names},
    }
    push = copy.deepcopy(close)
    push["command"]["prompt"] = "push the top drawer handle in"
    bowl_surface = copy.deepcopy(close)
    bowl_surface["command"]["prompt"] = "put the black bowl on top of the cabinet"

    assert same_vla_subgoal(push, close, language) is True
    assert same_vla_subgoal(bowl_surface, close, language) is False
    assert extract_vla_objects(push["command"]["prompt"], names, language) == {
        "fixture:top drawer"
    }
    assert extract_vla_objects(bowl_surface["command"]["prompt"], names, language) == {
        "object:akita black bowl",
        "fixture:cabinet top",
    }


def test_vla_object_matching_preserves_spatial_identity() -> None:
    names = ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1"]
    language = "stack the left bowl on the right bowl and place them in the tray"

    def step(prompt: str) -> dict[str, Any]:
        return {
            "command": {"action": "pi0_doubled", "prompt": prompt},
            "state": {"object_names": names},
        }

    assert (
        same_vla_subgoal(
            step("pick up the left bowl"), step("pick up the right bowl"), language
        )
        is False
    )
    assert (
        same_vla_subgoal(
            step("pick up the left bowl"),
            step("pick up the bowl on the left"),
            language,
        )
        is True
    )
    assert (
        same_vla_subgoal(
            step("pick up the black bowl"),
            step("pick up the left black bowl"),
            language,
        )
        is False
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "put the black bowl at the front on the plate",
            {"object:akita black bowl@front", "object:plate"},
        ),
        (
            "put the middle black bowl on the plate",
            {"object:akita black bowl@middle", "object:plate"},
        ),
        (
            "put the red mug on the left plate",
            {"object:red mug", "object:plate@left"},
        ),
        (
            "put the white mug on the right plate",
            {"object:white mug", "object:plate@right"},
        ),
    ],
)
def test_vla_object_spatial_identity_uses_the_qualified_noun(
    prompt: str, expected: set[str]
) -> None:
    names = ["akita_black_bowl_1", "plate_1", "red_mug_1", "white_mug_1"]

    assert extract_vla_objects(prompt, names, prompt) == expected


def test_vla_fixture_matching_covers_libero_90_action_objects() -> None:
    microwave = extract_vla_objects(
        "open the microwave door", [], "open the microwave door"
    )
    rack = extract_vla_objects(
        "place the wine in the wine rack", [], "place the wine in the wine rack"
    )
    front = extract_vla_objects(
        "put it in the front compartment of the caddy",
        [],
        "put it in the front compartment of the caddy",
    )
    left = extract_vla_objects(
        "put it in the left compartment of the caddy",
        [],
        "put it in the front compartment of the caddy",
    )

    assert microwave == {"fixture:door", "fixture:microwave"}
    assert rack == {"fixture:rack"}
    assert front == {"fixture:front compartment", "fixture:caddy"}
    assert left == {"fixture:left compartment", "fixture:caddy"}
    assert front != left


@pytest.mark.parametrize("name", ["basket", "tray", "caddy"])
def test_vla_movable_object_takes_precedence_over_same_named_fixture(
    name: str,
) -> None:
    prompt = f"put the item in the {name}"

    assert extract_vla_objects(prompt, [f"{name}_1"], prompt) == {f"object:{name}"}


def test_contact_skill_stopped_before_budget_is_not_a_weak_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop_early(steps: list[dict[str, Any]]) -> None:
        next(step for step in steps if step["step_idx"] == 2)["result"][
            "chunks_used"
        ] = 23

    cell = _build_cell(tmp_path, monkeypatch, 0, mutate=stop_early)

    episode = pair_recovery_episodes(cell)[0]
    assert [failure.step for failure in episode.failures] == [11]


def test_success_uses_shared_nested_tool_result_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def nest_success(steps: list[dict[str, Any]]) -> None:
        success = next(step for step in steps if step["step_idx"] == 14)
        success["result"] = {"log": {"result": success["result"]}}

    cell = _build_cell(tmp_path, monkeypatch, 2, reset=False, mutate=nest_success)

    assert pair_recovery_episodes(cell)[0].success_step == 14


def test_success_rejects_shared_libero_position_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def misleading_success(steps: list[dict[str, Any]]) -> None:
        failure = next(step for step in steps if step["step_idx"] == 3)
        success = next(step for step in steps if step["step_idx"] == 4)
        success["result"]["target_xyz"] = copy.deepcopy(failure["result"]["target_xyz"])
        success["result"]["final_dist_m"] = 0.02
        assert success["result"]["success"] is True
        assert success["result"]["diagnostics"]["tol"] == 0.012

    cell = _build_cell(tmp_path, monkeypatch, 1, mutate=misleading_success)

    assert pair_recovery_episodes(cell) == []


def test_writer_replaces_output_and_attaches_agent_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _build_cell(tmp_path, monkeypatch, 2)
    attempts = cell / "attempts"
    attempts.mkdir()
    (attempts / "attempt_1_failed.json").write_text(
        json.dumps({"agent_claim": "not pairing evidence"})
    )

    episodes = write_recovery_episodes(cell)
    manifest = json.loads((cell / "recovery_episodes.jsonl").read_text())

    assert manifest == episodes[0].to_manifest()
    assert manifest["agent_material"] == {
        "attempts": [
            {
                "attempt": 1,
                "failed": {"agent_claim": "not pairing evidence"},
            }
        ],
        "reset_commands": [{"step": 12, "reason": "retry approach"}],
    }
