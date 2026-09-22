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

"""Evidence-driven, bounded parameter adaptation for L1 recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping

from .diagnose import DiagnosisResult


@dataclass(frozen=True, slots=True)
class ParameterAdaptation:
    """A proposed argument delta and the evidence that justified it."""

    delta: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    adaptation_available: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy mutable mapping inputs into the frozen adaptation value."""
        object.__setattr__(self, "delta", dict(self.delta))
        object.__setattr__(self, "evidence", dict(self.evidence))


def _first(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


class ParameterAdapter:
    """Derive argument changes only from explicit numeric/target evidence.

    Pose targets are copied from evidence or applied as an explicit numeric
    vector delta.  Gripper changes likewise require an explicit target/opening
    or numeric delta; no default opening or distance is invented.  The runtime
    keeps state signatures independent of arguments and uses the ledger's
    per-parameter adaptation cap, so changing arguments cannot defeat
    no-progress termination.
    """

    @staticmethod
    def _argument_key(
        arguments: Mapping[str, Any], candidates: tuple[str, ...], fallback: str
    ) -> str:
        return next((key for key in candidates if key in arguments), fallback)

    @staticmethod
    def adapt(
        diagnosis: DiagnosisResult, arguments: Mapping[str, Any]
    ) -> ParameterAdaptation:
        """Derive one argument delta from explicit diagnosis evidence.

        Args:
            diagnosis: Structured failure diagnosis and its evidence channels.
            arguments: Arguments used for the failed tool call.

        Returns:
            An available pose, gripper, or approach-distance adaptation, or an
            unavailable adaptation when the evidence has no explicit target.
        """
        evidence = diagnosis.evidence
        pose = evidence.get("end_effector_pose", {}).get("value")
        if isinstance(pose, Mapping):
            target = _first(pose, ("target_pose", "desired_pose", "target", "pose"))
            delta_value = _first(
                pose, ("delta", "deviation", "pose_error", "position_error")
            )
            key = ParameterAdapter._argument_key(
                arguments, ("end_effector_pose", "ee_pose", "pose"), "pose"
            )
            if target is not None:
                return ParameterAdaptation(
                    {key: target},
                    "pose target copied from end-effector evidence",
                    True,
                    {
                        "signal": "end_effector_pose",
                        "source": "target_pose",
                        "old": arguments.get(key),
                        "new": target,
                    },
                )
            if isinstance(delta_value, (list, tuple)) and all(
                isinstance(item, Real) and not isinstance(item, bool)
                for item in delta_value
            ):
                current = arguments.get(key)
                if isinstance(current, (list, tuple)) and len(current) == len(
                    delta_value
                ):
                    new_value = type(current)(
                        left + right for left, right in zip(current, delta_value)
                    )
                    return ParameterAdaptation(
                        {key: new_value},
                        "pose delta applied from end-effector evidence",
                        True,
                        {
                            "signal": "end_effector_pose",
                            "source": "delta",
                            "old": current,
                            "new": new_value,
                        },
                    )

        opening_evidence = evidence.get("gripper_opening", {})
        opening = opening_evidence.get("value")
        text = evidence.get("transcript_text", {}).get("value")
        text = text.lower() if isinstance(text, str) else ""
        opening_data = opening if isinstance(opening, Mapping) else None
        opening_value = opening_data.get("value", opening) if opening_data else opening
        if (
            isinstance(opening_value, Real)
            and not isinstance(opening_value, bool)
            and opening_value <= 0
            and ("grasp" in text or "pick" in text)
        ):
            # A scalar opening is evidence of the failure, not a target.  A
            # target/step must be explicitly supplied to make a safe change.
            if isinstance(opening_data, Mapping):
                target = _first(
                    opening_data,
                    (
                        "target_opening",
                        "desired_opening",
                        "suggested_opening",
                        "opening",
                    ),
                )
                delta_value = _first(opening_data, ("delta", "opening_delta"))
            else:
                target = _first(
                    opening_evidence,
                    ("target_opening", "desired_opening", "suggested_opening"),
                )
                delta_value = _first(opening_evidence, ("delta", "opening_delta"))
            key = ParameterAdapter._argument_key(
                arguments, ("gripper_opening", "opening"), "gripper_opening"
            )
            if target is not None or isinstance(delta_value, Real):
                old = arguments.get(key)
                new_value = (
                    target
                    if target is not None
                    else (old + delta_value if isinstance(old, Real) else None)
                )
                if new_value is not None and new_value != old:
                    return ParameterAdaptation(
                        {key: new_value},
                        "gripper opening target/delta from grasp evidence",
                        True,
                        {
                            "signal": "gripper_opening",
                            "source": "explicit_target_or_delta",
                            "old": old,
                            "new": new_value,
                        },
                    )
            distance_evidence = evidence.get("approach_distance", {})
            distance_value = (
                distance_evidence.get("value")
                if isinstance(distance_evidence, Mapping)
                else None
            )
            distance_data = (
                distance_value
                if isinstance(distance_value, Mapping)
                else distance_evidence
            )
            if isinstance(distance_data, Mapping):
                target = _first(
                    distance_data, ("target", "target_distance", "desired_distance")
                )
                delta_value = _first(distance_data, ("delta", "distance_delta"))
                key = ParameterAdapter._argument_key(
                    arguments, ("approach_distance", "distance"), "approach_distance"
                )
                old = arguments.get(key)
                new_value = (
                    target
                    if target is not None
                    else (
                        old + delta_value
                        if isinstance(old, Real) and isinstance(delta_value, Real)
                        else None
                    )
                )
                if new_value is not None and new_value != old:
                    return ParameterAdaptation(
                        {key: new_value},
                        "approach distance target/delta from grasp evidence",
                        True,
                        {
                            "signal": "approach_distance",
                            "source": "explicit_target_or_delta",
                            "old": old,
                            "new": new_value,
                        },
                    )
        return ParameterAdaptation({}, "L1 无可用自适应证据", False, {})
