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

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rpent.recovery import (
    JsonlWriter,
    experience_gain,
    router_confusion_matrix,
    skill_reuse_rate,
)


class TestTrace:
    def test_all_metrics_are_computed_from_jsonl(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "decisions": [
                    {"event_id": "a", "level": "L1"},
                    {"event_id": "b", "level": "L2"},
                ],
                "oracle": [
                    {"event_id": "a", "level": "L1"},
                    {"event_id": "b", "level": "L1"},
                ],
                "pairs": [
                    {
                        "pair_id": "p",
                        "phase": "before",
                        "success": False,
                        "cost": {
                            "wall_clock_s": 9,
                            "turns": 8,
                            "upper_model_calls": 7,
                            "env_steps": 6,
                        },
                    },
                    {
                        "pair_id": "p",
                        "phase": "after",
                        "success": True,
                        "cost": {
                            "wall_clock_s": 5,
                            "turns": 4,
                            "upper_model_calls": 3,
                            "env_steps": 2,
                        },
                    },
                    {
                        "pair_id": "q",
                        "phase": "before",
                        "success": True,
                        "cost": {
                            "wall_clock_s": 8,
                            "turns": 6,
                            "upper_model_calls": 4,
                            "env_steps": 2,
                        },
                    },
                    {
                        "pair_id": "q",
                        "phase": "after",
                        "success": True,
                        "cost": {
                            "wall_clock_s": 6,
                            "turns": 4,
                            "upper_model_calls": 2,
                            "env_steps": 0,
                        },
                    },
                ],
                "reuse": [
                    {"retrieved": True, "adopted": True, "executed": True},
                    {"retrieved": True, "adopted": False, "executed": False},
                    {
                        "scoreable": False,
                        "retrieved": True,
                        "adopted": True,
                        "executed": True,
                    },
                ],
            }
            for name, rows in fixtures.items():
                with JsonlWriter(root / f"{name}.jsonl") as writer:
                    for row in rows:
                        writer.append(row)
            assert router_confusion_matrix(
                root / "decisions.jsonl", root / "oracle.jsonl"
            ) == {"L1": {"L1": 1, "L2": 1}}
            assert experience_gain(root / "pairs.jsonl") == {
                "EG_success": 0.5,
                "EG_cost": {
                    "wall_clock_s": 3.0,
                    "turns": 3.0,
                    "upper_model_calls": 3.0,
                    "env_steps": 3.0,
                },
            }
            assert skill_reuse_rate(root / "reuse.jsonl") == {
                "retrieved": 2,
                "adopted": 1,
                "executed": 1,
                "adoption_rate": 0.5,
                "execution_rate": 1.0,
                "reuse_rate": 0.5,
            }

    def test_experience_gain_rejects_missing_cost_dimension(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            with JsonlWriter(path) as writer:
                for phase in ("before", "after"):
                    writer.append(
                        {
                            "pair_id": "p",
                            "phase": phase,
                            "success": False,
                            "cost": {
                                "wall_clock_s": 1,
                                "turns": 1,
                                "upper_model_calls": 1,
                            },
                        }
                    )
            with pytest.raises(ValueError, match="missing cost dimension: env_steps"):
                experience_gain(path)

    def test_unscorable_decision_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with JsonlWriter(root / "decisions.jsonl") as writer:
                writer.append({"event_id": "infra", "level": "L2", "scoreable": False})
            with JsonlWriter(root / "oracle.jsonl") as writer:
                writer.append({"event_id": "infra", "level": "L2"})
            with pytest.raises(ValueError, match="unscorable"):
                router_confusion_matrix(root / "decisions.jsonl", root / "oracle.jsonl")
