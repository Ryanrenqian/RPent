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

"""Pair failed explore steps with later recovery successes.

The pairer depends on recovery events produced by observers after 98a617e; older
runs recorded an ``unknown`` failure family, so their strong-failure episodes
are missing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .libero_evidence import libero_tool_evidence, map_libero_evidence
from .persistence import JsonlWriter, read_jsonl
from .tool_result import (
    CONTACT_SKILL_SUCCESS_BY_TERMINATION,
    classify_tool_result_failure,
)

_VLA_TOOLS = frozenset({"pi0_pick", "pi0_doubled"})
_MOVE_TOOLS = frozenset({"move_to", "move_pose"})
# Supplied audit of LIBERO-90 BDDL filenames: these are the non-movable nouns
# used as action objects in task language.  Some scenes expose basket, tray, or
# caddy in ``object_names``; simulator objects take precedence in that case.
_FIXTURE_HEADS = frozenset(
    {
        "door",
        "shelf",
        "rack",
        "caddy",
        "compartment",
        "basket",
        "tray",
        "faucet",
        "knob",
        "burner",
        "drawer",
        "cabinet",
        "stove",
        "microwave",
    }
)
# These are the complete spatial words found in LIBERO-90 BDDL filenames.
_OBJECT_SPATIAL_MODIFIERS = ("left", "right", "middle", "front", "back")
_DRAWER_MODIFIERS = "|".join(
    (*_OBJECT_SPATIAL_MODIFIERS, "top", "bottom", "upper", "lower")
)

# User confirmed the multiplier on 2026-09-24.
_MOVE_TARGET_TOLERANCE_MULTIPLIER = 5


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


@dataclass(frozen=True, slots=True)
class EpisodeFailure:
    """One failure included in a recovery episode."""

    event_id: str | None
    step: int
    tool_id: str
    failure_family: str | None
    rule_id: str | None
    failure_source: str | None

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible failure reference.

        Returns:
            Manifest containing the failure fields used by the pairer.
        """
        return {
            "event_id": self.event_id,
            "step": self.step,
            "tool_id": self.tool_id,
            "failure_family": self.failure_family,
            "rule_id": self.rule_id,
            "failure_source": self.failure_source,
        }


@dataclass(frozen=True, slots=True)
class RecoveryEpisode:
    """Failures that share one later successful recovery step."""

    cell: str
    session: str
    level: str
    success_step: int
    success_tool_id: str
    success_command: Mapping[str, Any]
    failures: tuple[EpisodeFailure, ...]
    delta_steps: tuple[int, ...]
    cell_final_predicate: bool
    agent_material: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy nested mutable inputs and validate the pairing level."""
        if self.level not in {"intra_attempt", "cross_attempt"}:
            raise ValueError(f"invalid recovery episode level: {self.level}")
        object.__setattr__(self, "success_command", _json_copy(self.success_command))
        object.__setattr__(self, "agent_material", _json_copy(self.agent_material))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "delta_steps", tuple(self.delta_steps))

    def to_manifest(self) -> dict[str, Any]:
        """Return the complete JSON-compatible recovery episode.

        Returns:
            Manifest containing paired failures, recovery, and attachments.
        """
        return {
            "cell": self.cell,
            "session": self.session,
            "level": self.level,
            "success": {
                "step": self.success_step,
                "tool_id": self.success_tool_id,
                "command": _json_copy(self.success_command),
            },
            "failures": [failure.to_manifest() for failure in self.failures],
            "delta_steps": list(self.delta_steps),
            "cell_final_predicate": self.cell_final_predicate,
            "agent_material": _json_copy(self.agent_material),
        }


@dataclass(frozen=True, slots=True)
class _FailureCandidate:
    failure: EpisodeFailure
    record: Mapping[str, Any]
    attempt: int


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


def _object_aliases(object_name: str) -> tuple[str, ...]:
    tokens = _tokens(object_name)
    while tokens and tokens[-1].isdigit():
        tokens.pop()
    return tuple(" ".join(tokens[index:]) for index in range(len(tokens)))


def _object_modifier(prompt: str, start: int, end: int) -> str | None:
    modifiers = "|".join(_OBJECT_SPATIAL_MODIFIERS)
    prefix_match = re.search(rf"\b({modifiers})\s+$", prompt[:start])
    if prefix_match is not None:
        return prefix_match.group(1)
    suffix_match = re.match(
        rf"\s+(?:on|at) the ({modifiers})"
        r"(?=$|[,.]|\s+(?:on|in|into|onto|and|or|then|with|to|of)\b)",
        prompt[end:],
    )
    return suffix_match.group(1) if suffix_match is not None else None


def extract_vla_objects(
    prompt: str, object_names: Sequence[str], task_language: str
) -> frozenset[str]:
    """Extract deterministic VLA manipulation-object identities.

    Simulator objects are matched through suffix aliases of their normalized
    names, so ``akita_black_bowl_1`` matches ``black bowl`` or ``bowl``.  An
    adjacent LIBERO-90 spatial qualifier is part of the identity, such as
    ``object:akita black bowl@left``.  A simulator object wins over a fixture
    with the same head noun.
    LIBERO fixtures absent from ``object_names`` use syntax-preserving phrases:
    drawer ranks remain ``top drawer`` even before ``handle``, while a surface
    phrase such as ``top of the cabinet`` becomes ``cabinet top``.  Fixture
    phrases are retained only when their head noun also occurs in the cell's
    task language.

    Args:
        prompt: VLA command prompt to analyze.
        object_names: Simulator object identifiers from the step state.
        task_language: Initial cell instruction used to admit fixed fixtures.

    Returns:
        Canonical identities of objects manipulated by the prompt.
    """
    prompt_text = " ".join(_tokens(prompt))
    task_text = " ".join(_tokens(task_language))
    objects: set[str] = set()

    movable_fixture_heads: set[str] = set()
    object_mentions: list[tuple[int, int, str]] = []
    for object_name in object_names:
        aliases = _object_aliases(object_name)
        movable_fixture_heads.update(_FIXTURE_HEADS.intersection(aliases))
        for alias in aliases:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", prompt_text):
                object_mentions.append((match.start(), match.end(), aliases[0]))
    for start, end, object_identity in object_mentions:
        if any(
            other_start <= start
            and end <= other_end
            and other_end - other_start > end - start
            for other_start, other_end, _ in object_mentions
        ):
            continue
        identity = "object:" + object_identity
        modifier = _object_modifier(prompt_text, start, end)
        if modifier is not None:
            identity += "@" + modifier
        objects.add(identity)

    for match in re.finditer(
        rf"\b({_DRAWER_MODIFIERS}) drawer(?: handle)?\b", prompt_text
    ):
        identity = f"{match.group(1)} drawer"
        if re.search(rf"\b{re.escape(identity)}\b", task_text):
            objects.add("fixture:" + identity)

    spatial_modifiers = "|".join(_OBJECT_SPATIAL_MODIFIERS)
    qualified_fixture_heads: set[str] = set()
    for match in re.finditer(rf"\b({spatial_modifiers}) (compartment)\b", prompt_text):
        modifier, head = match.groups()
        if re.search(rf"\b{head}\b", task_text):
            objects.add(f"fixture:{modifier} {head}")
            qualified_fixture_heads.add(head)

    surface_pattern = (
        r"\b(?:on|onto) (?:the )?(top|bottom) of (?:the )?"
        rf"({'|'.join(sorted(_FIXTURE_HEADS))})\b"
    )
    surfaced_heads: set[str] = set()
    for match in re.finditer(surface_pattern, prompt_text):
        surface, head = match.groups()
        if re.search(rf"\b{head}\b", task_text):
            objects.add(f"fixture:{head} {surface}")
            surfaced_heads.add(head)

    owned_heads = {
        match.group(1)
        for match in re.finditer(
            rf"\bdrawer of (?:the )?({'|'.join(sorted(_FIXTURE_HEADS))})\b",
            prompt_text,
        )
    }
    for head in _FIXTURE_HEADS - {"drawer"} - movable_fixture_heads:
        if (
            head not in surfaced_heads
            and head not in owned_heads
            and head not in qualified_fixture_heads
            and re.search(rf"\b{head}\b", prompt_text)
            and re.search(rf"\b{head}\b", task_text)
        ):
            objects.add("fixture:" + head)

    return frozenset(objects)


def same_vla_subgoal(
    first: Mapping[str, Any], second: Mapping[str, Any], task_language: str
) -> bool:
    """Return whether two VLA step records manipulate the same objects.

    Args:
        first: First serialized step record.
        second: Second serialized step record.
        task_language: Initial cell instruction for fixed-fixture matching.

    Returns:
        Whether both commands use the same VLA tool and object identities.
    """
    first_command = first.get("command")
    second_command = second.get("command")
    if not isinstance(first_command, Mapping) or not isinstance(
        second_command, Mapping
    ):
        return False
    if first_command.get("action") != second_command.get("action"):
        return False
    if first_command.get("action") not in _VLA_TOOLS:
        return False
    first_state = first.get("state")
    second_state = second.get("state")
    if not isinstance(first_state, Mapping) or not isinstance(second_state, Mapping):
        return False
    first_names = first_state.get("object_names")
    second_names = second_state.get("object_names")
    if not isinstance(first_names, list) or not isinstance(second_names, list):
        return False
    first_prompt = first_command.get("prompt")
    second_prompt = second_command.get("prompt")
    if not isinstance(first_prompt, str) or not isinstance(second_prompt, str):
        return False
    first_objects = extract_vla_objects(first_prompt, first_names, task_language)
    second_objects = extract_vla_objects(second_prompt, second_names, task_language)
    # Conservatively keep qualified and unqualified mentions distinct: missing a
    # pair is safer than pairing actions that may target different instances.
    return bool(first_objects) and first_objects == second_objects


def _is_success(record: Mapping[str, Any]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    failure_source, _ = classify_tool_result_failure(result)
    if failure_source is not None:
        return False
    evidence = libero_tool_evidence(map_libero_evidence(result))
    return not any(
        isinstance(item.get("value"), Mapping) and item["value"].get("reached") is False
        for item in evidence.values()
    )


def _same_subgoal(
    failure: Mapping[str, Any], success: Mapping[str, Any], task_language: str
) -> bool:
    failure_command = failure.get("command")
    success_command = success.get("command")
    if not isinstance(failure_command, Mapping) or not isinstance(
        success_command, Mapping
    ):
        return False
    action = failure_command.get("action")
    if action != success_command.get("action"):
        return False
    if action in _VLA_TOOLS:
        return same_vla_subgoal(failure, success, task_language)
    if action not in _MOVE_TOOLS:
        return False
    failure_result = failure.get("result")
    success_result = success.get("result")
    if not isinstance(failure_result, Mapping) or not isinstance(
        success_result, Mapping
    ):
        return False
    diagnostics = failure_result.get("diagnostics")
    failure_target = failure_result.get("target_xyz")
    success_target = success_result.get("target_xyz")
    if not isinstance(diagnostics, Mapping):
        return False
    tolerance = diagnostics.get("tol")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not isinstance(failure_target, (list, tuple))
        or not isinstance(success_target, (list, tuple))
        or len(failure_target) != len(success_target)
    ):
        return False
    return (
        math.dist(failure_target, success_target)
        <= _MOVE_TARGET_TOLERANCE_MULTIPLIER * tolerance
    )


def _rule_id(event: Mapping[str, Any]) -> str | None:
    diagnosis = event.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        return None
    rule = diagnosis.get("rule_id")
    if isinstance(rule, Mapping) and isinstance(rule.get("value"), str):
        return rule["value"]
    return None


def _weak_failure(record: Mapping[str, Any]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    diagnostics = result.get("diagnostics")
    return (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("mode") == CONTACT_SKILL_SUCCESS_BY_TERMINATION
        and result.get("chunks_used") == result.get("max_chunks")
        and result.get("chunks_used") is not None
        and result.get("terminated") is False
    )


def _load_steps(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload, Mapping) else payload
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        raise ValueError(f"{path} does not contain a StepRecord list")
    return sorted(steps, key=lambda step: step["step_idx"])


def _attempts(steps: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    attempt = 1
    result: dict[int, int] = {}
    for step in steps:
        step_index = step["step_idx"]
        result[step_index] = attempt
        command = step.get("command")
        if isinstance(command, Mapping) and command.get("action") == "reset":
            attempt += 1
    return result


def _task_language(steps: Sequence[Mapping[str, Any]]) -> str:
    if not steps:
        return ""
    extras = steps[0].get("extras")
    if isinstance(extras, Mapping) and isinstance(extras.get("task_language"), str):
        return extras["task_language"]
    return ""


def _failure_candidates(
    steps: Sequence[Mapping[str, Any]],
    attempts: Mapping[int, int],
    event_records: Sequence[Mapping[str, Any]],
) -> list[_FailureCandidate]:
    by_step = {step["step_idx"]: step for step in steps}
    candidates: list[_FailureCandidate] = []
    strong_steps: set[int] = set()
    for record in event_records:
        event = record.get("event")
        if not isinstance(event, Mapping) or event.get("failure_family") == "unknown":
            continue
        step_index = event.get("step")
        step = by_step.get(step_index)
        if step is None or not isinstance(event.get("tool_id"), str):
            continue
        metadata = event.get("metadata")
        source = (
            metadata.get("failure_source")
            if isinstance(metadata, Mapping)
            and isinstance(metadata.get("failure_source"), str)
            else None
        )
        failure = EpisodeFailure(
            event_id=event.get("event_id"),
            step=step_index,
            tool_id=event["tool_id"],
            failure_family=event["failure_family"],
            rule_id=_rule_id(event),
            failure_source=source,
        )
        candidates.append(_FailureCandidate(failure, step, attempts[step_index]))
        strong_steps.add(step_index)

    for step in steps:
        step_index = step["step_idx"]
        command = step.get("command")
        if (
            step_index not in strong_steps
            and isinstance(command, Mapping)
            and isinstance(command.get("action"), str)
            and _weak_failure(step)
        ):
            failure = EpisodeFailure(
                event_id=None,
                step=step_index,
                tool_id=command["action"],
                failure_family=None,
                rule_id=None,
                failure_source="budget_exhausted_contact",
            )
            candidates.append(_FailureCandidate(failure, step, attempts[step_index]))
    return sorted(candidates, key=lambda candidate: candidate.failure.step)


def _agent_material(
    cell: Path, failure_attempts: set[int], delta: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    attempt_material = []
    for attempt in sorted(failure_attempts):
        path = cell / "attempts" / f"attempt_{attempt}_failed.json"
        if path.is_file():
            attempt_material.append(
                {"attempt": attempt, "failed": json.loads(path.read_text())}
            )
    resets = []
    for step in delta:
        command = step.get("command")
        if isinstance(command, Mapping) and command.get("action") == "reset":
            resets.append(
                {
                    "step": step["step_idx"],
                    "reason": command.get("reason"),
                }
            )
    return {"attempts": attempt_material, "reset_commands": resets}


def pair_recovery_episodes(cell_dir: str | Path) -> list[RecoveryEpisode]:
    """Pair all eligible failures and recoveries in one explore cell.

    Args:
        cell_dir: Explore cell containing sessions and optional attempt notes.

    Returns:
        Recovery episodes ordered by session and successful step.
    """
    cell = Path(cell_dir)
    sessions = sorted((cell / "sessions").glob("session_*"))
    session_steps = {
        session: _load_steps(session / "states.json") for session in sessions
    }
    final_predicate = any(
        step.get("terminated") is True
        for steps in session_steps.values()
        for step in steps
    )
    episodes: list[RecoveryEpisode] = []
    for session, steps in session_steps.items():
        attempts = _attempts(steps)
        language = _task_language(steps)
        event_path = session / "recovery_events.jsonl"
        event_records = read_jsonl(event_path) if event_path.is_file() else []
        failures = _failure_candidates(steps, attempts, event_records)
        grouped: dict[tuple[int, str], list[_FailureCandidate]] = {}
        for failure in failures:
            intra = next(
                (
                    step
                    for step in steps
                    if step["step_idx"] > failure.failure.step
                    and attempts[step["step_idx"]] == failure.attempt
                    and _is_success(step)
                    and _same_subgoal(failure.record, step, language)
                ),
                None,
            )
            success = intra
            level = "intra_attempt"
            if success is None and final_predicate:
                success = next(
                    (
                        step
                        for step in steps
                        if attempts[step["step_idx"]] == failure.attempt + 1
                        and _is_success(step)
                        and _same_subgoal(failure.record, step, language)
                    ),
                    None,
                )
                level = "cross_attempt"
            if success is not None:
                grouped.setdefault((success["step_idx"], level), []).append(failure)

        by_step = {step["step_idx"]: step for step in steps}
        for (success_index, level), grouped_failures in sorted(grouped.items()):
            success = by_step[success_index]
            earliest_failure = min(item.failure.step for item in grouped_failures)
            delta = [
                step
                for step in steps
                if earliest_failure < step["step_idx"] < success_index
            ]
            command = success["command"]
            episodes.append(
                RecoveryEpisode(
                    cell=str(cell),
                    session=session.name,
                    level=level,
                    success_step=success_index,
                    success_tool_id=command["action"],
                    success_command=command,
                    failures=tuple(item.failure for item in grouped_failures),
                    delta_steps=tuple(step["step_idx"] for step in delta),
                    cell_final_predicate=final_predicate,
                    agent_material=_agent_material(
                        cell, {item.attempt for item in grouped_failures}, delta
                    ),
                )
            )
    return episodes


def write_recovery_episodes(cell_dir: str | Path) -> list[RecoveryEpisode]:
    """Pair and write one cell's episodes to ``recovery_episodes.jsonl``.

    Args:
        cell_dir: Explore cell to read and update.

    Returns:
        Recovery episodes written to the cell.
    """
    cell = Path(cell_dir)
    episodes = pair_recovery_episodes(cell)
    destination = cell / "recovery_episodes.jsonl"
    destination.unlink(missing_ok=True)
    with JsonlWriter(destination) as writer:
        for episode in episodes:
            writer.append(episode.to_manifest())
    return episodes
