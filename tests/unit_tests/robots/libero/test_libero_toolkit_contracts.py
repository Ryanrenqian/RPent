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

"""Offline contracts for the LIBERO toolkit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.libero import robot_spec, toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig
from rpent.tools.toolkit import Toolkit, _is_readonly
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EVALUATION_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "move_to",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
    "move_pose",
    "view_camera_meta",
    "segment",
    "back_project",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={
            "memory_dir": str(memory_dir),
            "reference_tag": "libero_10_task_t0_s0",
        },
        task_desc={},
    )


def test_toolkit_factory_configures_memory_access_by_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_toolkit(
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: Any,
        memory: MemoryManager,
        recovery_goal: str | None,
        mode: str,
        attempts_per_session: int,
        state_output_dir: Path | str | None,
    ) -> SimpleNamespace:
        values = {
            "runtime_kwargs": runtime_kwargs,
            "dashboard_events": dashboard_events,
            "memory": memory,
            "recovery_goal": recovery_goal,
            "mode": mode,
            "attempts_per_session": attempts_per_session,
            "state_output_dir": state_output_dir,
        }
        captured.append(values)
        return SimpleNamespace(**values)

    monkeypatch.setattr(toolkit, "LiberoToolkit", fake_toolkit)
    memory_dir = tmp_path / "libero-memory"
    config = _run_config(memory_dir)

    evaluation = robot_spec.get_toolkit(
        runtime_kwargs={"env": "evaluation"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    exploration = robot_spec.get_toolkit(
        runtime_kwargs={"env": "exploration"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=2,
        state_output_dir=tmp_path / "state",
    )

    assert evaluation.memory.root == memory_dir.resolve()
    assert exploration.memory.root == memory_dir.resolve()
    evaluation_write = evaluation.memory.get_common_tool_bindings()["write_text_file"][
        1
    ]
    exploration_write = exploration.memory.get_common_tool_bindings()[
        "write_text_file"
    ][1]
    own_draft = memory_dir / "_internal" / "inbox" / config.recipe_tag / "draft.md"
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        evaluation_write(str(own_draft), "draft")
    assert exploration_write(str(own_draft), "draft")["bytes_written"] == 5
    assert captured[0]["mode"] == "evaluation"
    assert captured[1]["mode"] == "exploration"
    assert captured[1]["attempts_per_session"] == 2
    assert captured[0]["recovery_goal"] == "libero_10_task_t0_s0"
    assert captured[1]["recovery_goal"] == "libero_10_task_t0_s0"


def test_recovery_goal_is_stable_across_seeds_for_the_same_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_goals: list[str | None] = []

    def fake_toolkit(
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: Any,
        memory: MemoryManager,
        recovery_goal: str | None,
        mode: str,
        attempts_per_session: int,
        state_output_dir: Path | str | None,
    ) -> SimpleNamespace:
        captured_goals.append(recovery_goal)
        return SimpleNamespace(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=dashboard_events,
            memory=memory,
            recovery_goal=recovery_goal,
            mode=mode,
            attempts_per_session=attempts_per_session,
            state_output_dir=state_output_dir,
        )

    def args_for(seed: int) -> SimpleNamespace:
        return SimpleNamespace(
            suite="libero_10_task",
            task=4,
            seed=seed,
            planner=None,
            molmo_endpoint=None,
            explore=False,
            explore_sessions=3,
            collect_flywheel_data=False,
            memory_profile="hf",
            memory_dir=None,
            output_dir=tmp_path / f"run-{seed}",
        )

    first = robot_spec._parse_config(args_for(0))
    second = robot_spec._parse_config(args_for(9))
    monkeypatch.setattr(toolkit, "LiberoToolkit", fake_toolkit)
    for config in (first, second):
        robot_spec.get_toolkit(
            runtime_kwargs={},
            dashboard_events=NullDashboardEventSink(),
            config=config,
        )

    assert first.recipe_tag != second.recipe_tag
    assert first.prompt_vars["reference_tag"] == "10_task_t4_s0"
    assert second.prompt_vars["reference_tag"] == first.prompt_vars["reference_tag"]
    assert captured_goals == ["10_task_t4_s0", "10_task_t4_s0"]


def test_toolkit_modes_construct_with_fake_primitives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "LiberoPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    evaluation = toolkit.LiberoToolkit(
        runtime_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "evaluation-memory"),
        mode="evaluation",
        state_output_dir=tmp_path / "evaluation",
    )
    exploration = toolkit.LiberoToolkit(
        runtime_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(
            tmp_path / "exploration-memory",
            memory_access="inbox_write",
            inbox_cell_tag="offline-cell",
        ),
        mode="exploration",
        attempts_per_session=3,
        state_output_dir=tmp_path / "exploration",
    )

    assert _tool_names(evaluation) == EVALUATION_TOOLS
    assert _tool_names(exploration) == EVALUATION_TOOLS | {"reset"}
    assert _readonly_names(evaluation) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "segment",
        "back_project",
    }
    assert _readonly_names(exploration) == _readonly_names(evaluation)
    assert len(dumped) == 2
    assert all(
        instance.reset_calls == 1 for instance in fake_single_arm_primitives.instances
    )
    assert all(
        instance.recording_started for instance in fake_single_arm_primitives.instances
    )
    assert all(
        callable(instance.kwargs["check_cancelled"])
        for instance in fake_single_arm_primitives.instances
    )

    refused = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "first attempt"}
    )
    assert refused.result["error"] == "finish refused"
    assert refused.is_finish is False

    exploration.get_env_state = lambda *, command, result, elapsed_s: dict(result)
    assert (
        exploration.execute_tool("reset", {"reason": "new approach"}).result["attempt"]
        == 2
    )
    assert (
        exploration.execute_tool("reset", {"reason": "third approach"}).result[
            "attempt"
        ]
        == 3
    )
    allowed = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "budget spent"}
    )
    assert allowed.is_finish is True
