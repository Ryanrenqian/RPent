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

from typing import Any

import numpy as np

from robots.libero.tools import LiberoPrimitives


class _StaticLiberoEnv:
    terminated = False
    truncated = False

    @staticmethod
    def raw_obs() -> dict[str, Any]:
        return {"robot0_eef_quat": [0.0, 0.0, 0.0, 1.0]}


def _static_primitives() -> LiberoPrimitives:
    primitives = object.__new__(LiberoPrimitives)
    primitives.env = _StaticLiberoEnv()
    primitives._last_obs_eef_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    primitives._step_env = lambda action: None
    return primitives


def test_libero_motion_results_publish_their_call_thresholds() -> None:
    move = _static_primitives().move_to([0.2, 0.0, 0.0], max_steps=1, tol=0.05)
    assert move["success"] is False
    assert move["diagnostics"]["tol"] == 0.05

    move_pose = _static_primitives().move_pose(
        [0.2, 0.0, 0.0], max_steps=1, tol=0.05, ori_tol=0.1
    )
    assert move_pose["success"] is False
    assert move_pose["diagnostics"] == {"tol": 0.05, "ori_tol": 0.1}
