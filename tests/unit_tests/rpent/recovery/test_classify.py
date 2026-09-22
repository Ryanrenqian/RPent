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

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rpent.recovery import (
    CellSignals,
    CellVerdict,
    InfraProtocolClassifier,
    signals_from_cell_dir,
)
from rpent.recovery.classify import (
    INFRA_PATTERNS,
    NORMAL_END_RE,
    ROBOT_TOOL_RE,
    TIMEOUT_RE,
)


def signals(**changes):
    values = {
        "predicate": False,
        "robot_tool_calls": 1,
        "transcript": {"finish": {"status": "failure"}},
        "log_text": "[tool<-] move_to",
        "protocol_timeout": False,
        "normal_end": False,
        "cell_nonempty": True,
    }
    values.update(changes)
    return CellSignals(**values)


class TestCellClassification:
    def test_infrastructure_failures_do_not_enter_diagnoser(self):
        for name, text in (
            ("no_mcp_tools", "[agent] transcript:"),
            ("413", "413 Payload Too Large"),
            ("upstream_disconnect", "stream disconnected before completion"),
            ("relay_capacity", "service is busy"),
        ):
            record = InfraProtocolClassifier().classify(
                signals(
                    predicate=True,
                    robot_tool_calls=0 if name == "no_mcp_tools" else 1,
                    transcript={"finish": {"status": "failure"}}
                    if name == "no_mcp_tools"
                    else None,
                    log_text=text,
                )
            )
            assert record.verdict == CellVerdict.UNSCORABLE
            assert record.infra_bucket == name
            assert record.unscorable_reason == name
            assert not record.scoreable
            assert record.to_failure_event_input() is None

    def test_completed_success_ignores_nonterminal_413_marker(self):
        record = InfraProtocolClassifier().classify(
            signals(
                predicate=True,
                robot_tool_calls=42,
                transcript={"finish": {"status": "success"}},
                log_text="[tool<-] move_to\n413 Payload Too Large\n[tool<-] finish\n[agent] transcript:",
                normal_end=True,
            )
        )
        assert record.verdict == CellVerdict.SUCCESS
        assert record.scoreable
        assert record.infra_bucket is None
        assert record.unscorable_reason is None

    def test_protocol_outcomes_are_scored(self):
        for timed_out, normal_end, reason, report in (
            (True, False, "budget_exhausted", "no_finish_timeout"),
            (False, True, "agent_gave_up", "no_finish_clean_end"),
        ):
            record = InfraProtocolClassifier().classify(
                signals(
                    transcript={"finish": None},
                    protocol_timeout=timed_out,
                    normal_end=normal_end,
                )
            )
            assert record.verdict == CellVerdict.SCORED_PROTOCOL_FAILURE
            assert record.scored_reason == reason
            assert record.self_report == report
            assert record.scoreable
            assert record.to_failure_event_input()["scored_reason"] == reason

    def test_predicate_overrides_self_report_and_missing_predicate_is_failure(self):
        classifier = InfraProtocolClassifier()
        success = classifier.classify(signals(predicate=True))
        assert success.verdict == CellVerdict.SUCCESS
        assert not success.self_report_agreement
        assert success.to_failure_event_input() is None
        assert (
            classifier.classify(signals(predicate=None)).verdict
            == CellVerdict.TASK_FAILURE
        )
        assert classifier.classify(signals()).verdict == CellVerdict.TASK_FAILURE
        assert success.evidence["robot_tool_calls"] == {
            "source": "log.ROBOT_TOOL_RE",
            "value": 1,
        }

    def test_incomplete_or_unparseable_cell_is_unscorable(self):
        cases = (
            ({"transcript": None}, "no_finish"),
            ({"transcript": {"finish": None}}, "no_finish"),
            (
                {"transcript": None, "transcript_unparseable": True},
                "unparseable_transcript",
            ),
            ({"transcript": None, "log_text": None}, "no_log"),
            (
                {"transcript": None, "log_text": None, "cell_nonempty": False},
                "not_started",
            ),
        )
        for changes, expected_reason in cases:
            record = InfraProtocolClassifier().classify(signals(**changes))
            assert record.verdict == CellVerdict.UNSCORABLE
            assert record.unscorable_reason == expected_reason
            assert record.infra_bucket is None
            assert record.to_failure_event_input() is None

    def test_cell_directory_reads_states_transcript_and_log(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "libero_10_task" / "task_2" / "seed_3"
            cell.mkdir(parents=True)
            (cell / "states.json").write_text(
                json.dumps({"steps": [{"terminated": False}, {"terminated": True}]})
            )
            (cell / "transcript_10_task_t2_s3.json").write_text(
                json.dumps({"finish": {"status": "failure"}})
            )
            (root / "logs").mkdir()
            (root / "logs" / "libero_10_task_t2_s3.log").write_text(
                "[tool<-] move_to\n[agent] transcript:\nCodex SDK timed out after 5000s"
            )
            parsed = signals_from_cell_dir(cell)
            assert parsed.predicate
            assert parsed.robot_tool_calls == 1
            assert parsed.protocol_timeout
            assert parsed.normal_end
            assert (
                InfraProtocolClassifier().classify(parsed).verdict
                == CellVerdict.SUCCESS
            )

    def test_no_mcp_cell_directory_never_produces_failure_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "libero_10_task" / "task_1" / "seed_0"
            cell.mkdir(parents=True)
            (cell / "states.json").write_text('{"steps": [{"terminated": false}]}')
            (cell / "transcript_10_task_t1_s0.json").write_text(
                '{"finish": {"status": "failure"}}'
            )
            (root / "logs").mkdir()
            (root / "logs" / "libero_10_task_t1_s0.log").write_text(
                "[agent] transcript: done"
            )
            record = InfraProtocolClassifier().classify(signals_from_cell_dir(cell))
            assert record.infra_bucket == "no_mcp_tools"
            assert record.verdict == CellVerdict.UNSCORABLE
            assert record.to_failure_event_input() is None

    def test_archived_cell_uses_original_seed_for_transcript_and_suffix_for_log(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "libero_10_task" / "task_1" / "seed_0_invalid_413_20260922"
            cell.mkdir(parents=True)
            (cell / "transcript_10_task_t1_s0.json").write_text('{"finish": null}')
            (root / "logs").mkdir()
            (
                root / "logs" / "libero_10_task_t1_s0_invalid_413_20260922.log"
            ).write_text("[tool<-] move_to\n413 Payload Too Large")
            parsed = signals_from_cell_dir(cell)
            assert parsed.robot_tool_calls == 1
            assert InfraProtocolClassifier().classify(parsed).infra_bucket == "413"

    def test_cell_directory_distinguishes_missing_and_unparseable_transcript(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "libero_10_task" / "task_1" / "seed_0"
            cell.mkdir(parents=True)
            (root / "logs").mkdir()
            log = root / "logs" / "libero_10_task_t1_s0.log"
            log.write_text("[tool<-] move_to")
            missing = signals_from_cell_dir(cell)
            assert not missing.transcript_unparseable
            assert (
                InfraProtocolClassifier().classify(missing).unscorable_reason
                == "no_finish"
            )
            (cell / "transcript_10_task_t1_s0.json").write_text("{")
            malformed = signals_from_cell_dir(cell)
            assert malformed.transcript_unparseable
            assert (
                InfraProtocolClassifier().classify(malformed).unscorable_reason
                == "unparseable_transcript"
            )

    def test_infra_patterns_cannot_drift_from_reproduction_snapshot(self):
        source = Path(__file__).parent / "fixtures" / "summarize_patterns.txt"
        tree = ast.parse(source.read_text())
        original = next(
            (
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    (
                        isinstance(target, ast.Name) and target.id == "INFRA_PATTERNS"
                        for target in node.targets
                    )
                )
            )
        )
        expected = [
            (ast.literal_eval(pair.elts[0]), ast.literal_eval(pair.elts[1].args[0]))
            for pair in original.elts
        ]
        assert [(bucket, regex.pattern) for bucket, regex in INFRA_PATTERNS] == expected
        assert {"no_mcp_tools", *(bucket for bucket, _ in INFRA_PATTERNS)} == {
            "no_mcp_tools",
            *(bucket for bucket, _ in expected),
        }
        local_regexes = {
            "TIMEOUT_RE": TIMEOUT_RE,
            "NORMAL_END_RE": NORMAL_END_RE,
            "ROBOT_TOOL_RE": ROBOT_TOOL_RE,
        }
        for name, local_regex in local_regexes.items():
            original_regex = next(
                (
                    node.value
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        (
                            isinstance(target, ast.Name) and target.id == name
                            for target in node.targets
                        )
                    )
                )
            )
            assert local_regex.pattern == ast.literal_eval(original_regex.args[0])
