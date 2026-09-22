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

import time

import pytest

from rpent.recovery import (
    BudgetLedger,
    BudgetLimits,
    SkillPlaybook,
    SkillRuntime,
    SkillStep,
    ToolRegistry,
    ToolSpec,
)
from rpent.recovery.budget import CellBudget


class TestBudgetAndRuntime:
    @staticmethod
    def skill(*tool_ids):
        return SkillPlaybook(
            skill_id="timed",
            name="Timed",
            goal="pick",
            trigger_labels=frozenset({"x"}),
            diagnosis="d",
            steps=tuple(
                (SkillStep(str(i), "p", tool_id) for i, tool_id in enumerate(tool_ids))
            ),
            verification_checks=("v",),
        )

    def test_wall_clock_budget_exhausts_on_real_sleep(self):
        registry = ToolRegistry()

        def slow_failure(args, state):
            time.sleep(0.02)
            raise RuntimeError("miss")

        registry.register(ToolSpec("slow", "Slow", "sleeps"), slow_failure)
        result = SkillRuntime(
            registry,
            ledger=BudgetLedger(limits=BudgetLimits(max_wall_clock_s=0.01)),
            recovery_loop=True,
        ).execute(self.skill("slow"), episode_id="ep-clock")
        assert result.decision.termination_reason == "budget_exhausted"
        assert result.ledger.attempts == 1
        assert result.ledger.wall_clock_s > 0.01
        assert result.tool_results[0].metadata["wall_clock_s"] > 0
        assert result.decision.cost["wall_clock_s"] == result.ledger.wall_clock_s

    def test_alternating_signatures_eventually_give_up(self):
        ledger = BudgetLedger(max_no_progress=2)
        reasons = [
            ledger.record_attempt(state_signature=value)
            for value in ("A", "B", "A", "B", "A")
        ]
        assert reasons[:4] == [None] * 4
        assert reasons[4] == "no_progress"

    def test_default_wall_clock_is_below_planner_timeout(self):
        limits = BudgetLimits()
        assert limits.max_wall_clock_s < 5000
        assert (
            limits.max_wall_clock_s,
            limits.max_turns,
            limits.max_env_steps,
            limits.max_upper_model_calls,
        ) == (1000.0, 20, 2000, 20)

    def test_recovery_budget_scales_all_dimensions_and_validates_fraction(self):
        limits = BudgetLimits.from_cell_budget(CellBudget(), 0.35)
        assert (
            limits.max_wall_clock_s,
            limits.max_turns,
            limits.max_env_steps,
            limits.max_upper_model_calls,
        ) == (1750.0, 35, 3500, 35)
        for fraction in (0, -0.1, 1.0, 1.1):
            with pytest.raises(
                ValueError,
                match="recovery wall clock must be strictly below planner timeout",
            ):
                BudgetLimits.from_cell_budget(CellBudget(), fraction)

    def test_parameter_adaptation_counts_are_bounded_per_parameter(self):
        ledger = BudgetLedger(max_parameter_adaptations=2)
        assert ledger.record_parameter_adaptation("pose") is None
        assert ledger.record_parameter_adaptation("pose") is None
        assert (
            ledger.record_parameter_adaptation("pose") == "parameter_adaptation_limit"
        )
        assert ledger.record_parameter_adaptation("gripper_opening") is None
        assert ledger.record_parameter_adaptation("gripper_opening") is None
        assert (
            ledger.record_parameter_adaptation("gripper_opening")
            == "parameter_adaptation_limit"
        )
        assert ledger.parameter_adaptations == {"pose": 2, "gripper_opening": 2}
