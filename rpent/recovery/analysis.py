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

"""Offline metrics computed exclusively from persisted JSONL traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .persistence import read_jsonl


def router_confusion_matrix(
    decisions: str | Path, oracle_labels: str | Path
) -> dict[str, dict[str, int]]:
    """Count oracle level to router level, joining on unique event IDs.

    Args:
        decisions: JSONL path containing router decisions.
        oracle_labels: JSONL path containing expected recovery levels.

    Returns:
        Counts grouped first by oracle level and then by routed level.

    Raises:
        ValueError: If records are unscorable, duplicated, or unmatched.
    """
    labels: dict[str, str] = {}
    for record in read_jsonl(oracle_labels):
        event_id = record["event_id"]
        if event_id in labels:
            raise ValueError(f"duplicate oracle event_id: {event_id}")
        labels[event_id] = record["level"]
    matrix: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for record in read_jsonl(decisions):
        if record.get("scoreable") is False:
            raise ValueError("unscorable decision must not enter router analysis")
        event_id = record["event_id"]
        if event_id in seen:
            raise ValueError(f"duplicate decision event_id: {event_id}")
        seen.add(event_id)
        if event_id not in labels:
            raise ValueError(f"missing oracle label for {event_id}")
        row = matrix.setdefault(labels[event_id], {})
        level = record["level"]
        row[level] = row.get(level, 0) + 1
    return matrix


def diagnoser_confusion_matrix(
    predictions: str | Path, labels: str | Path
) -> dict[str, dict[str, int]]:
    """Count labelled failure family -> diagnosed family by unique event_id.

    The fixture currently used by this function is synthetic, not human
    annotation.  Both inputs must be scoreable; duplicate or unmatched event
    IDs fail loudly so an audit cannot silently change its denominator.

    Args:
        predictions: JSONL path containing diagnosed failure families.
        labels: JSONL path containing expected failure families.

    Returns:
        Counts grouped first by labelled family and then by diagnosed family.

    Raises:
        ValueError: If records are unscorable, duplicated, or unmatched.
    """
    expected: dict[str, str] = {}
    for record in read_jsonl(labels):
        if (
            record.get("scoreable") is not True
            or record.get("provenance") == "unscorable"
        ):
            raise ValueError(
                "unscorable or unmarked label must not enter diagnoser analysis"
            )
        event_id = record["event_id"]
        if event_id in expected:
            raise ValueError(f"duplicate diagnoser label event_id: {event_id}")
        expected[event_id] = record["failure_family"]
    matrix: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for record in read_jsonl(predictions):
        if record.get("scoreable") is not True:
            raise ValueError(
                "unscorable or unmarked diagnosis must not enter diagnoser analysis"
            )
        event_id = record["event_id"]
        if event_id in seen:
            raise ValueError(f"duplicate diagnosis event_id: {event_id}")
        seen.add(event_id)
        if event_id not in expected:
            raise ValueError(f"missing diagnoser label for {event_id}")
        row = matrix.setdefault(expected[event_id], {})
        family = record["failure_family"]
        row[family] = row.get(family, 0) + 1
    if set(expected) != seen:
        missing = sorted(set(expected) - seen)
        raise ValueError(f"missing diagnosis for {missing[0]}")
    return matrix


def experience_gain(records: str | Path) -> dict[str, float | dict[str, float]]:
    """Mean paired after-before success and before-after cost by dimension.

    Each JSONL record needs pair_id, phase (before/after), success (bool), and
    all four cost dimensions. Missing or duplicate partners fail rather than
    bias means.

    Args:
        records: JSONL path containing paired before-and-after outcomes.

    Returns:
        Mean success gain and mean cost reduction for each cost dimension.

    Raises:
        ValueError: If pairs, phases, success values, or costs are invalid.
    """
    cost_dimensions = ("wall_clock_s", "turns", "upper_model_calls", "env_steps")
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for record in read_jsonl(records):
        if record.get("scoreable") is False:
            continue
        pair = pairs.setdefault(record["pair_id"], {})
        phase = record["phase"]
        if phase not in {"before", "after"} or phase in pair:
            raise ValueError(f"invalid or duplicate phase for pair {record['pair_id']}")
        pair[phase] = record
    if not pairs or any(set(pair) != {"before", "after"} for pair in pairs.values()):
        raise ValueError("experience gain requires complete before/after pairs")
    success_gain = 0
    cost_gain = dict.fromkeys(cost_dimensions, 0.0)
    for pair in pairs.values():
        before, after = pair["before"], pair["after"]
        if not isinstance(before["success"], bool) or not isinstance(
            after["success"], bool
        ):
            raise ValueError("success must be a boolean")
        success_gain += int(after["success"]) - int(before["success"])
        for dimension in cost_dimensions:
            try:
                costs = (before["cost"][dimension], after["cost"][dimension])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"missing cost dimension: {dimension}") from exc
            if any(
                isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0
                for cost in costs
            ):
                raise ValueError(f"{dimension} must be a non-negative number")
            cost_gain[dimension] += costs[0] - costs[1]
    return {
        "EG_success": success_gain / len(pairs),
        "EG_cost": {
            dimension: gain / len(pairs) for dimension, gain in cost_gain.items()
        },
    }


def skill_reuse_rate(records: str | Path) -> dict[str, float | int]:
    """Adoption rate = adopted/retrieved; execution rate = executed/adopted.

    Each JSONL record is one candidate with retrieved, adopted, executed bools.
    Retrieval includes candidates returned but not adopted; zero denominators
    yield 0.0. Executed candidates must also be adopted and retrieved.

    Args:
        records: JSONL path containing skill reuse flags.

    Returns:
        Reuse counts and adoption, execution, and overall reuse rates.

    Raises:
        ValueError: If reuse flags are not booleans or violate their ordering.
    """
    retrieved = adopted = executed = 0
    for record in read_jsonl(records):
        if record.get("scoreable") is False:
            continue
        flags = (record["retrieved"], record["adopted"], record["executed"])
        if (
            not all(isinstance(flag, bool) for flag in flags)
            or (flags[2] and not flags[1])
            or (flags[1] and not flags[0])
        ):
            raise ValueError("invalid skill reuse flags")
        retrieved += flags[0]
        adopted += flags[1]
        executed += flags[2]
    return {
        "retrieved": retrieved,
        "adopted": adopted,
        "executed": executed,
        "adoption_rate": adopted / retrieved if retrieved else 0.0,
        "execution_rate": executed / adopted if adopted else 0.0,
        "reuse_rate": executed / retrieved if retrieved else 0.0,
    }
