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

import json

import pytest

from rpent.recovery import (
    ExecutionEvent,
    FailureEvent,
    RecoveryAction,
    RecoveryDecision,
    RecoveryLevel,
    ToolGapEvent,
)


class TestEvolutionCore:
    def test_events_are_typed_and_immutable(self):
        event = ExecutionEvent("ep", "pick object", 0, state={"x": 1})
        assert event.state["x"] == 1
        with pytest.raises(ValueError):
            ExecutionEvent("", "pick object", 0)


class TestTrace:
    def test_event_and_decision_round_trip(self):
        base = {
            "episode_id": "ep",
            "goal": "pick",
            "step": 2,
            "event_id": "fixed",
            "timestamp": 123.5,
            "completed_subgoals": ("open",),
            "state": {"joints": [1, 2]},
            "metadata": {"confidence": 0.5},
        }
        events = (
            ExecutionEvent(**base),
            FailureEvent(
                **base,
                failure_family="grasp",
                diagnosis={"pose": [1, 2]},
                retryable=True,
                parameter_issue=True,
            ),
            ToolGapEvent(**base, failure_family="gap", missing_capability="insert"),
        )
        for event in events:
            manifest = event.to_manifest()
            assert json.loads(json.dumps(manifest)) == manifest
            assert type(event).from_manifest(manifest) == event
            assert manifest["completed_subgoals"] == ["open"]
        decision = RecoveryDecision(
            RecoveryLevel.L1,
            RecoveryAction.ADAPT,
            "pose",
            event_id="fixed",
            episode_id="ep",
            step=2,
            cost={
                "turns": 2,
                "wall_clock_s": 4.5,
                "upper_model_calls": 1,
                "env_steps": 3,
            },
            termination_reason="budget_exhausted",
        )
        assert RecoveryDecision.from_manifest(decision.to_manifest()) == decision
        assert decision.to_manifest()["level"] == "L1"
        for constructor, kwargs in (
            (ExecutionEvent, {**base, "event_id": ""}),
            (
                RecoveryDecision,
                {
                    "level": RecoveryLevel.L0,
                    "action": RecoveryAction.RETRY,
                    "reason": "x",
                    "cost": {"env_steps": -1},
                },
            ),
        ):
            with pytest.raises(ValueError):
                constructor(**kwargs)
