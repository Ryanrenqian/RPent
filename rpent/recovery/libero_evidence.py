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

    target = action_result.get("target_xyz")
    final = action_result.get("final_eef_pos")
    if isinstance(target, (list, tuple)) and isinstance(final, (list, tuple)):
        if len(target) == len(final):
            pose: dict[str, Any] = {
                "reachable": action_result.get("success"),
                "pose_error": [
                    target_value - final_value
                    for target_value, final_value in zip(target, final)
                ],
            }
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
