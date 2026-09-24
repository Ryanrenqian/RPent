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

"""Translate LIBERO observations into the generic recovery evidence shape."""

from __future__ import annotations

from typing import Any, Mapping


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _evidence(source: str, value: Any) -> dict[str, Any]:
    return {"source": source, "value": value}


def map_libero_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract recovery evidence from one LIBERO tool result.

    The pose residual is useful for a servo with a steady-state tracking
    offset: ``ParameterAdapter`` adds it to the next commanded pose.  It is
    not a general reachability oracle.  If the requested target is itself
    unreachable because of kinematic limits, collision, or occlusion,
    continuing in the same direction can move farther away until the bounded
    L1 budget is exhausted; that failure belongs to L2 or requires object
    position evidence.

    Args:
        result: One LIBERO tool result, including its optional nested state.

    Returns:
        Generic recovery evidence fields.  An empty mapping is returned when
        the result contains no supported LIBERO fields.
    """
    mapped: dict[str, Any] = {}
    action_result: Mapping[str, Any] = result
    log = result.get("log")
    if isinstance(log, Mapping) and isinstance(log.get("result"), Mapping):
        action_result = log["result"]

    state = result.get("state", action_result.get("state"))
    state_mapping = state if isinstance(state, Mapping) else {}
    diagnostics = action_result.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    tool_name = action_result.get("name")

    final_dist = action_result.get("final_dist_m")
    tolerance = diagnostics.get("tol")
    if (
        tool_name in {"move_to", "move_pose"}
        and _number(final_dist)
        and _number(tolerance)
    ):
        mapped["libero_position"] = _evidence(
            f"LIBERO {tool_name} result: final_dist_m vs diagnostics.tol",
            {
                "reached": final_dist < tolerance,
                "final_dist_m": final_dist,
                "tol": tolerance,
            },
        )

    orientation_tolerance = diagnostics.get("ori_tol")
    final_pitch = action_result.get("final_pitch")
    if (
        tool_name == "move_pose"
        and action_result.get("success") is False
        and "libero_position" in mapped
        and mapped["libero_position"]["value"]["reached"] is True
        and _number(final_pitch)
        and _number(orientation_tolerance)
    ):
        mapped["libero_orientation"] = _evidence(
            "LIBERO move_pose result: success=False while final_dist_m < "
            "diagnostics.tol; "
            "final_pitch and diagnostics.ori_tol",
            {
                "reached": False,
                "final_pitch": final_pitch,
                "ori_tol": orientation_tolerance,
            },
        )

    descent = diagnostics.get("descent_m")
    descent_threshold = diagnostics.get("descent_thresh")
    if tool_name == "pick" and _number(descent) and _number(descent_threshold):
        mapped["libero_pick_descent"] = _evidence(
            "LIBERO pi0_pick result: diagnostics.descent_m vs "
            "diagnostics.descent_thresh",
            {
                "reached": descent >= descent_threshold,
                "descent_m": descent,
                "descent_thresh": descent_threshold,
            },
        )

    opening = action_result.get("min_gripper_opening")
    open_threshold = diagnostics.get("gripper_open_thresh")
    closed_threshold = diagnostics.get("gripper_closed_thresh")
    if (
        tool_name == "pick"
        and _number(opening)
        and _number(open_threshold)
        and _number(closed_threshold)
    ):
        mapped["libero_pick_close"] = _evidence(
            "LIBERO pi0_pick result: min_gripper_opening vs "
            "diagnostics.gripper_open_thresh/gripper_closed_thresh",
            {
                "reached": open_threshold <= opening < closed_threshold,
                "min_gripper_opening": opening,
                "gripper_open_thresh": open_threshold,
                "gripper_closed_thresh": closed_threshold,
            },
        )

    peak_lift = action_result.get("peak_lift_m")
    lift_threshold = diagnostics.get("lift_thresh")
    if tool_name == "pick" and _number(peak_lift) and _number(lift_threshold):
        mapped["libero_pick_lift"] = _evidence(
            "LIBERO pi0_pick result: peak_lift_m vs diagnostics.lift_thresh",
            {
                "reached": peak_lift >= lift_threshold,
                "peak_lift_m": peak_lift,
                "lift_thresh": lift_threshold,
            },
        )

    target = action_result.get("target_xyz")
    final = action_result.get("final_eef_pos")
    if isinstance(target, (list, tuple)) and isinstance(final, (list, tuple)):
        if len(target) == len(final):
            pose: dict[str, Any] = {
                "pose_error": [
                    target_value - final_value
                    for target_value, final_value in zip(target, final)
                ],
            }
            if tool_name in {"move_to", "move_pose"} and "libero_position" in mapped:
                pose["reachable"] = mapped["libero_position"]["value"]["reached"]
            elif tool_name not in {"move_to", "move_pose"}:
                pose["reachable"] = action_result.get("success")
            current = state_mapping.get("robot0_eef_pos")
            if isinstance(current, (list, tuple)):
                pose["current_pose"] = list(current)
            mapped["end_effector_pose"] = pose
    elif isinstance(state_mapping.get("robot0_eef_pos"), (list, tuple)):
        mapped["end_effector_pose"] = {
            "current_pose": list(state_mapping["robot0_eef_pos"])
        }

    gripper_qpos = state_mapping.get("robot0_gripper_qpos")
    if isinstance(gripper_qpos, (list, tuple)) and len(gripper_qpos) >= 2:
        mapped["gripper_opening"] = sum(abs(value) for value in gripper_qpos[:2])

    return mapped


def libero_tool_evidence(mapped: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Select tool-returned LIBERO predicate evidence from mapped observations."""
    return {key: value for key, value in mapped.items() if key.startswith("libero_")}
