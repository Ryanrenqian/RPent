# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline snapshot RPC contracts for the LIBERO adapter."""

from __future__ import annotations

import numpy as np
import pytest

from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade


class _VectorEnv:
    def __init__(self) -> None:
        self.states = [np.array([1.0, 2.0], dtype=np.float64)]
        self.observation = {"marker": np.array(["old"], dtype=object)}
        self.set_calls: list[tuple[object, int]] = []

    def get_sim_state(self):
        return list(self.states)

    def set_init_state(self, init_state, id):
        assert isinstance(init_state, (list, tuple))
        assert len(init_state) == 1
        assert np.asarray(init_state[0]).shape == (2,)
        assert id == 0
        self.set_calls.append((init_state, id))
        self.states[0] = np.asarray(init_state[0], dtype=np.float64)
        self.observation = {"marker": np.array(["new"], dtype=object)}
        return np.stack([self.observation])


class _ServerEnv:
    def __init__(self) -> None:
        self.env = _VectorEnv()
        self.current_raw_obs = [{"marker": np.array(["old"], dtype=object)}]


def _facade() -> LiberoEnvFacade:
    return LiberoEnvFacade(_ServerEnv(), meta={})


def test_facade_snapshot_rpc_uses_single_env_shapes_and_refreshes_raw_obs():
    facade = _facade()

    np.testing.assert_array_equal(facade.get_sim_state(), [1.0, 2.0])
    assert "env.get_sim_state" in facade._readonly_methods
    assert "env.set_sim_state" not in facade._readonly_methods

    restored = facade.set_sim_state(np.array([3.0, 4.0]))

    assert restored["marker"].tolist() == ["new"]
    assert facade.raw_obs()["marker"].tolist() == ["new"]
    assert facade._env.env.set_calls[0][1] == 0


class _Rpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict, float | None]] = []
        self.state = np.array([1.0, 2.0], dtype=np.float64)

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs or {}, timeout_s))
        if method == "env.get_env_meta":
            return {}
        if method == "env.reset":
            return {"marker": "reset"}, {}
        if method == "env.get_sim_state":
            return self.state.copy()
        if method == "env.set_sim_state":
            self.state = np.asarray(args[0], dtype=np.float64)
            return {"marker": "restored"}
        if method == "env.step":
            return {"marker": "stepped"}, 0.0, False, False, {}
        raise AssertionError(method)


def test_client_restores_pre_termination_cache():
    rpc = _Rpc()
    client = LiberoEnvClient(rpc, expected_meta={})
    client.last_obs = {"marker": "before"}
    state = client.get_sim_state()
    client.terminated = True
    client.truncated = True
    client.last_obs = {"marker": "after"}

    client.set_sim_state(state)

    assert client.terminated is False
    assert client.truncated is False
    assert client.last_obs == {"marker": "before"}
    client.step(np.zeros(7))


def test_client_restores_post_termination_cache():
    rpc = _Rpc()
    client = LiberoEnvClient(rpc, expected_meta={})
    client.terminated = True
    client.truncated = True
    client.last_obs = {"marker": "terminated"}
    state = client.get_sim_state()
    client.terminated = False
    client.truncated = False
    client.last_obs = {"marker": "changed"}

    client.set_sim_state(state)

    assert client.terminated is True
    assert client.truncated is True
    assert client.last_obs == {"marker": "terminated"}
    with pytest.raises(AssertionError, match="episode signaled"):
        client.step(np.zeros(7))


def test_client_rejects_unbookmarked_state_before_remote_mutation():
    client = LiberoEnvClient(_Rpc(), expected_meta={})
    with pytest.raises(KeyError, match="not captured"):
        client.set_sim_state(np.array([9.0, 9.0]))
