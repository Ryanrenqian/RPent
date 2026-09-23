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

"""LIBERO env client that forwards calls over an RPC transport.

Lives in :mod:`robots.libero` because the methods exposed
here (``raw_obs`` / ``render_camera`` / ``get_camera_meta`` / …)
reference LIBERO-specific obs dict keys and camera names. The generic
transport layer lives in :mod:`rpent.utils.rpc.socket_rpc`.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class LiberoEnvClient(BaseEnvClient):
    """Remote implementation of the LIBERO env protocol."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.render_camera": 120.0,
    }

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict,
        return_all_frames: bool = False,
    ):
        self.return_all_frames = return_all_frames
        self.terminated = False
        self.truncated = False
        self._sim_state_cache: dict[bytes, tuple[bool, bool, Any]] = {}
        super().__init__(client, expected_meta=expected_meta)

    def check_done(self, term, trunc) -> None:
        self.terminated |= bool(np.asarray(term).any())
        self.truncated |= bool(np.asarray(trunc).any())

    def reset(self) -> tuple[dict, Any]:
        self.last_obs, info = super().reset()
        self.terminated = False
        self.truncated = False
        return self.last_obs, info

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        assert not (self.terminated or self.truncated), (
            "env.step called after the episode signaled term/trunc"
        )
        ret = super().step(action)
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def chunk_step(
        self, actions, *, return_all_frames: bool | None = None
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Run an action chunk in one RPC. Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``.

        ``obs`` is ``list[Obs]`` when ``return_all_frames`` is True
        (one entry per chunk step), otherwise the final ``Obs`` dict.
        Terminated / truncated have shape ``[chunk_size]`` after the
        server strips the env dim.
        """
        assert not (self.terminated or self.truncated), (
            "env.chunk_step called after the episode signaled term/trunc"
        )
        if return_all_frames is None:
            return_all_frames = self.return_all_frames
        ret = super().chunk_step(
            actions,
            return_all_frames=return_all_frames,
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def raw_obs(self) -> dict:
        return self._client.call("env.raw_obs", timeout_s=self._TIMEOUT_S["default"])

    @staticmethod
    def _state_key(state) -> bytes:
        """Build a stable local key for a flattened simulator state."""
        array = np.asarray(state, dtype=np.float64)
        return array.tobytes()

    def get_sim_state(self) -> np.ndarray:
        """Return and locally bookmark the current flattened simulator state."""
        state = np.asarray(
            self._client.call(
                "env.get_sim_state", timeout_s=self._TIMEOUT_S["default"]
            ),
            dtype=np.float64,
        ).copy()
        self._sim_state_cache[self._state_key(state)] = (
            self.terminated,
            self.truncated,
            copy.deepcopy(self.last_obs),
        )
        return state

    def set_sim_state(self, state) -> dict[str, Any]:
        """Restore a simulator snapshot and its client-side episode cache.

        Args:
            state: A state previously returned by :meth:`get_sim_state`.

        Returns:
            The restored raw observation returned by the environment server.

        Raises:
            KeyError: If the state was not captured by this client, because its
                terminated/truncated cache cannot be reconstructed safely.
        """
        key = self._state_key(state)
        cached = self._sim_state_cache.get(key)
        if cached is None:
            raise KeyError(
                "sim state was not captured by this client; cache rollback is undefined"
            )
        restored = self._client.call(
            "env.set_sim_state",
            args=(np.asarray(state, dtype=np.float64),),
            timeout_s=self._TIMEOUT_S["default"],
        )
        self.terminated, self.truncated, self.last_obs = (
            cached[0],
            cached[1],
            copy.deepcopy(cached[2]),
        )
        return restored

    def capture_sim_state(self):
        """Capture simulator state and client episode-cache state together."""
        return self.get_sim_state()

    def restore_sim_state(self, snapshot) -> None:
        """Restore a snapshot captured by :meth:`capture_sim_state`."""
        self.set_sim_state(snapshot)

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return super().render_camera(
            camera_name, height=height, width=width, depth=depth
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return super().get_camera_meta(camera_name, height=height, width=width)
