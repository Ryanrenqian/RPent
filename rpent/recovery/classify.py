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

"""Scoreable cell outcomes using the RPent reproduction's artifact rules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

# Keep these expressions in sync with deploy/rpent_gpt55/summarize.py.
INFRA_PATTERNS = (
    ("413", re.compile(r"413 Payload Too Large")),
    ("upstream_disconnect", re.compile(r"stream disconnected before completion")),
    (
        "relay_capacity",
        re.compile(
            r"Concurrency limit exceeded|currently overloaded|service is busy|Too many pending requests"
        ),
    ),
)
TIMEOUT_RE = re.compile(r"Codex SDK timed out after (\d+)s|interrupted duration=\d+")
NORMAL_END_RE = re.compile(r"\[agent\] transcript:")
ROBOT_TOOL_RE = re.compile(
    r"\[tool<-\] (view_env_state|move_to|pi0_pick|pi0_doubled|back_project|segment"
    r"|set_gripper|release|rotate_wrist|rotate_pitch|move_pose|finish)\b"
)


class CellVerdict(str, Enum):
    """Scoreability and outcome categories for one protocol cell."""

    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    SCORED_PROTOCOL_FAILURE = "scored_protocol_failure"
    UNSCORABLE = "unscorable"


@dataclass(frozen=True, slots=True)
class CellSignals:
    """Artifact-derived signals used to classify one protocol cell."""

    predicate: bool | None
    robot_tool_calls: int
    transcript: Mapping[str, Any] | None
    log_text: str | None
    protocol_timeout: bool
    normal_end: bool
    cell_nonempty: bool
    transcript_unparseable: bool = False

    def __post_init__(self) -> None:
        """Validate the observed robot tool-call count."""
        if self.robot_tool_calls < 0:
            raise ValueError("robot_tool_calls must be non-negative")


@dataclass(frozen=True, slots=True)
class CellRecord:
    """Classification result with its scoreability and source evidence."""

    verdict: CellVerdict
    infra_bucket: str | None
    unscorable_reason: str | None
    scored_reason: str | None
    predicate: bool | None
    self_report: str | None
    self_report_agreement: bool | None
    scoreable: bool
    evidence: Mapping[str, Any]

    def to_failure_event_input(self) -> dict[str, Any] | None:
        """Build diagnoser input for a scoreable non-success outcome.

        Returns:
            Failure evidence, or ``None`` for successes and unscorable cells.
        """
        if not self.scoreable or self.verdict == CellVerdict.SUCCESS:
            return None
        return {
            "predicate": self.predicate,
            "self_report": self.self_report,
            "self_report_agreement": self.self_report_agreement,
            "scored_reason": self.scored_reason,
            "evidence": dict(self.evidence),
        }


class InfraProtocolClassifier:
    """Classify cell artifacts using deterministic protocol precedence."""

    def classify(self, signals: CellSignals) -> CellRecord:
        """Classify a normalized set of cell signals.

        Args:
            signals: Predicate, transcript, log, and completeness signals.

        Returns:
            A scoreable task/protocol verdict or an unscorable infrastructure
            record with source evidence.
        """
        data = signals.transcript
        finish = data.get("finish") if data is not None else None
        status = finish.get("status") if isinstance(finish, Mapping) else None
        bucket = None
        reason = None
        infra_marker = None
        unscorable_reason = None

        if signals.transcript_unparseable:
            unscorable_reason = "unparseable_transcript"
        elif signals.log_text is not None and signals.robot_tool_calls == 0:
            bucket = "no_mcp_tools"
            unscorable_reason = bucket
        elif finish is not None:
            pass
        elif data is not None and signals.protocol_timeout:
            status = "no_finish_timeout"
            reason = "budget_exhausted"
        elif data is not None and signals.log_text is not None and signals.normal_end:
            status = "no_finish_clean_end"
            reason = "agent_gave_up"
        elif signals.cell_nonempty:
            if signals.log_text is None:
                unscorable_reason = "no_log"
            else:
                for name, pattern in INFRA_PATTERNS:
                    match = pattern.search(signals.log_text)
                    if match:
                        bucket = name
                        infra_marker = match.group()
                        unscorable_reason = name
                        break
                if unscorable_reason is None:
                    unscorable_reason = "no_finish"
        else:
            unscorable_reason = "not_started"

        complete = data is not None and (finish is not None or reason is not None)
        scoreable = unscorable_reason is None and complete
        agreement = None
        if signals.predicate is not None and status is not None:
            agreement = signals.predicate == (status == "success")
        evidence = {
            "predicate": {
                "source": "states.json.steps[*].terminated",
                "value": signals.predicate,
            },
            "robot_tool_calls": {
                "source": "log.ROBOT_TOOL_RE",
                "value": signals.robot_tool_calls,
            },
            "self_report": {"source": "transcript.finish.status", "value": status},
            "protocol_timeout": {
                "source": "log.TIMEOUT_RE",
                "value": signals.protocol_timeout,
            },
            "normal_end": {"source": "log.NORMAL_END_RE", "value": signals.normal_end},
            "cell_nonempty": {
                "source": "cell directory or log",
                "value": signals.cell_nonempty,
            },
            "transcript_unparseable": {
                "source": "transcript JSON parsing",
                "value": signals.transcript_unparseable,
            },
            "infra_marker": {"source": "log_text", "value": infra_marker},
        }
        if not scoreable:
            verdict = CellVerdict.UNSCORABLE
        elif signals.predicate is True:
            verdict = CellVerdict.SUCCESS
        elif reason is not None:
            verdict = CellVerdict.SCORED_PROTOCOL_FAILURE
        else:
            verdict = CellVerdict.TASK_FAILURE
        return CellRecord(
            verdict=verdict,
            infra_bucket=bucket,
            unscorable_reason=unscorable_reason,
            scored_reason=reason
            if verdict == CellVerdict.SCORED_PROTOCOL_FAILURE
            else None,
            predicate=signals.predicate,
            self_report=status,
            self_report_agreement=agreement,
            scoreable=scoreable,
            evidence=evidence,
        )


def signals_from_cell_dir(path: str | Path) -> CellSignals:
    """Read a seed directory and its sibling suite log.

    Args:
        path: ``<root>/<suite>/task_N/seed_N`` directory to inspect.

    Returns:
        Signals parsed from states, transcript, directory, and log artifacts.

    Raises:
        ValueError: If the task or seed directory naming is invalid.
    """
    cell = Path(path)
    task = cell.parent.name
    seed = cell.name
    suite = cell.parent.parent.name
    if not task.startswith("task_") or not seed.startswith("seed_"):
        raise ValueError("expected <root>/<suite>/task_N/seed_N cell directory")
    seed_number = seed[5:].split("_", 1)[0]
    if not seed_number.isdecimal() or not task[5:].isdecimal():
        raise ValueError("task and seed indices must be numeric")
    suffix = seed[len(f"seed_{seed_number}") :]
    if suffix and not suffix.startswith("_invalid_"):
        raise ValueError("unrecognized archived seed directory")
    log = (
        cell.parent.parent.parent
        / "logs"
        / f"{suite}_t{task[5:]}_s{seed_number}{suffix}.log"
    )
    try:
        text = log.read_text(errors="replace") if log.is_file() else None
    except OSError:
        text = None

    predicate = None
    try:
        states = json.loads((cell / "states.json").read_text())
        steps = states.get("steps") or []
        if steps:
            predicate = any(bool(step.get("terminated")) for step in steps)
    except (OSError, json.JSONDecodeError):
        pass

    short = suite.removeprefix("libero_10_")
    transcript_path = cell / f"transcript_10_{short}_t{task[5:]}_s{seed_number}.json"
    transcript_unparseable = False
    try:
        transcript = json.loads(transcript_path.read_text())
        if not isinstance(transcript, dict):
            transcript = None
            transcript_unparseable = True
    except json.JSONDecodeError:
        transcript = None
        transcript_unparseable = True
    except OSError:
        transcript = None

    return CellSignals(
        predicate=predicate,
        robot_tool_calls=len(ROBOT_TOOL_RE.findall(text or "")),
        transcript=transcript,
        log_text=text,
        protocol_timeout=bool(TIMEOUT_RE.search(text or "")),
        normal_end=bool(NORMAL_END_RE.search(text or "")),
        cell_nonempty=(cell.is_dir() and any(cell.iterdir())) or log.is_file(),
        transcript_unparseable=transcript_unparseable,
    )
