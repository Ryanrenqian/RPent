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

"""Experiential skill playbooks and deterministic retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .persistence import SKILL_MANIFEST_SCHEMA, read_manifest, write_manifest


@dataclass(frozen=True, slots=True)
class SkillStep:
    """A playbook step references a tool by ID; it does not contain source."""

    step_id: str
    purpose: str
    tool_id: str
    argument_bindings: Mapping[str, Any] = field(default_factory=dict)
    checks: tuple[str, ...] = ()
    on_failure: str = "route"

    def __post_init__(self) -> None:
        """Validate identifiers and copy step bindings and checks."""
        for field_name in ("step_id", "purpose", "tool_id", "on_failure"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        object.__setattr__(self, "argument_bindings", dict(self.argument_bindings))
        object.__setattr__(self, "checks", tuple(self.checks))


@dataclass(frozen=True, slots=True)
class SkillValidation:
    """Evidence identifier and the four independent skill validation gates."""

    evidence_id: str
    replay_passed: bool
    perturbation_passed: bool
    held_out_passed: bool
    regression_passed: bool
    notes: str = ""

    @property
    def passed(self) -> bool:
        """Whether every validation gate passed."""
        return all(
            (
                self.replay_passed,
                self.perturbation_passed,
                self.held_out_passed,
                self.regression_passed,
            )
        )

    def to_manifest(self) -> dict[str, Any]:
        """Serialize the validation gates.

        Returns:
            JSON-compatible validation manifest.
        """
        return {
            "evidence_id": self.evidence_id,
            "replay_passed": self.replay_passed,
            "perturbation_passed": self.perturbation_passed,
            "held_out_passed": self.held_out_passed,
            "regression_passed": self.regression_passed,
            "notes": self.notes,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "SkillValidation":
        """Restore validation gates from a manifest.

        Args:
            payload: Serialized validation fields.

        Returns:
            Restored validation result.
        """
        return cls(
            evidence_id=str(payload["evidence_id"]),
            replay_passed=bool(payload["replay_passed"]),
            perturbation_passed=bool(payload["perturbation_passed"]),
            held_out_passed=bool(payload["held_out_passed"]),
            regression_passed=bool(payload["regression_passed"]),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class SkillPlaybook:
    """Reusable experience for solving a class of failures with tools."""

    skill_id: str
    name: str
    goal: str
    trigger_labels: frozenset[str]
    diagnosis: str
    steps: tuple[SkillStep, ...]
    verification_checks: tuple[str, ...]
    fallback_branches: tuple[str, ...] = ()
    applicability: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: SkillValidation | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        """Validate playbook fields and freeze mutable container inputs."""
        for field_name in ("skill_id", "name", "goal", "diagnosis", "version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.steps:
            raise ValueError("skill must contain at least one step")
        if not self.verification_checks:
            raise ValueError("skill must contain verification checks")
        object.__setattr__(self, "trigger_labels", frozenset(self.trigger_labels))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "verification_checks", tuple(self.verification_checks))
        object.__setattr__(self, "fallback_branches", tuple(self.fallback_branches))
        object.__setattr__(self, "applicability", dict(self.applicability))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def verified(self) -> bool:
        """Whether attached validation exists and passes every gate."""
        return self.validation is not None and self.validation.passed

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this playbook and its optional validation.

        Returns:
            JSON-compatible skill manifest without tool executors.
        """
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "goal": self.goal,
            "trigger_labels": sorted(self.trigger_labels),
            "diagnosis": self.diagnosis,
            "steps": [
                {
                    "step_id": step.step_id,
                    "purpose": step.purpose,
                    "tool_id": step.tool_id,
                    "argument_bindings": dict(step.argument_bindings),
                    "checks": list(step.checks),
                    "on_failure": step.on_failure,
                }
                for step in self.steps
            ],
            "verification_checks": list(self.verification_checks),
            "fallback_branches": list(self.fallback_branches),
            "applicability": dict(self.applicability),
            "provenance": dict(self.provenance),
            "validation": self.validation.to_manifest() if self.validation else None,
            "version": self.version,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "SkillPlaybook":
        """Restore a skill playbook from a manifest.

        Args:
            payload: Serialized playbook fields.

        Returns:
            Validated playbook with reconstructed steps and validation.
        """
        validation = payload.get("validation")
        return cls(
            skill_id=str(payload["skill_id"]),
            name=str(payload["name"]),
            goal=str(payload["goal"]),
            trigger_labels=frozenset(payload.get("trigger_labels", ())),
            diagnosis=str(payload["diagnosis"]),
            steps=tuple(
                SkillStep(
                    step_id=str(step["step_id"]),
                    purpose=str(step["purpose"]),
                    tool_id=str(step["tool_id"]),
                    argument_bindings=dict(step.get("argument_bindings", {})),
                    checks=tuple(step.get("checks", ())),
                    on_failure=str(step.get("on_failure", "route")),
                )
                for step in payload["steps"]
            ),
            verification_checks=tuple(payload["verification_checks"]),
            fallback_branches=tuple(payload.get("fallback_branches", ())),
            applicability=dict(payload.get("applicability", {})),
            provenance=dict(payload.get("provenance", {})),
            validation=SkillValidation.from_manifest(validation)
            if validation
            else None,
            version=str(payload.get("version", "1")),
        )


class SkillBankError(ValueError):
    """Raised for invalid skill-bank registration or lookup operations."""


class ExperienceSkillBank:
    """Persistent-facing skill store with deterministic label-overlap retrieval."""

    def __init__(self) -> None:
        """Initialize an empty skill bank."""
        self._skills: dict[str, SkillPlaybook] = {}

    def promote(self, skill: SkillPlaybook) -> None:
        """Add a fully validated, uniquely identified playbook.

        Args:
            skill: Playbook to promote.

        Raises:
            SkillBankError: If validation is incomplete or the ID already exists.
        """
        if not skill.verified:
            raise SkillBankError("only fully validated skills may enter the bank")
        if skill.skill_id in self._skills:
            raise SkillBankError(f"skill already registered: {skill.skill_id}")
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> SkillPlaybook:
        """Return a playbook by ID.

        Args:
            skill_id: Registered playbook identifier.

        Returns:
            Matching playbook.

        Raises:
            SkillBankError: If the identifier is unknown.
        """
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise SkillBankError(f"unknown skill: {skill_id}") from exc

    def list_skills(self) -> tuple[SkillPlaybook, ...]:
        """Return registered playbooks in insertion order."""
        return tuple(self._skills.values())

    def manifest(self) -> dict[str, Any]:
        """Serialize the bank to a schema-tagged manifest."""
        return {
            "schema": SKILL_MANIFEST_SCHEMA,
            "skills": [skill.to_manifest() for skill in self._skills.values()],
        }

    def save_manifest(self, path: str | Path) -> None:
        """Persist the bank atomically.

        Args:
            path: Destination manifest path.
        """
        write_manifest(path, self.manifest())

    @classmethod
    def load_manifest(cls, path: str | Path) -> "ExperienceSkillBank":
        """Load and validate a persisted skill bank.

        Args:
            path: Source manifest path.

        Returns:
            Reconstructed skill bank.

        Raises:
            ValueError: If the manifest's ``skills`` field is not a list.
        """
        payload = read_manifest(path, SKILL_MANIFEST_SCHEMA)
        skills = payload.get("skills")
        if not isinstance(skills, list):
            raise ValueError("skill manifest 'skills' must be a list")
        bank = cls()
        for item in skills:
            bank.promote(SkillPlaybook.from_manifest(item))
        return bank

    def retrieve(
        self, *, goal: str, labels: set[str] | frozenset[str]
    ) -> SkillPlaybook | None:
        """Retrieve the deterministic best label match for an exact goal.

        Args:
            goal: Exact playbook goal to match.
            labels: Failure labels used for overlap ranking.

        Returns:
            Best verified playbook, or ``None`` when no labels overlap.

        Raises:
            ValueError: If ``goal`` is empty.
        """
        if not goal.strip():
            raise ValueError("goal must not be empty")
        query_labels = frozenset(labels)
        ranked: list[tuple[int, int, str, SkillPlaybook]] = []
        for skill in self._skills.values():
            if skill.goal != goal:
                continue
            overlap = len(skill.trigger_labels & query_labels)
            if overlap:
                ranked.append(
                    (-overlap, -len(skill.trigger_labels), skill.skill_id, skill)
                )
        return min(ranked)[3] if ranked else None
