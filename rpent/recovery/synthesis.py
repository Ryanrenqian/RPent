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

"""Tool synthesis boundary and a deterministic local mock backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .handoff import RPentHandoff
from .tools import ToolError, ToolExecutor, ToolRegistry, ToolSpec
from .verification import Case, Check, ToolVerifier, VerificationReport


class ToolSynthesisError(RuntimeError):
    """Raised when a synthesizer cannot produce a candidate tool."""


@dataclass(frozen=True, slots=True)
class CandidateTool:
    """Synthesized descriptor, executor binding, and provenance metadata."""

    spec: ToolSpec
    executor: ToolExecutor
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the executor and copy candidate provenance."""
        if not callable(self.executor):
            raise TypeError("candidate executor must be callable")
        object.__setattr__(self, "provenance", dict(self.provenance))


class ToolSynthesizer(Protocol):
    """Protocol implemented by candidate tool synthesis backends."""

    def synthesize(self, handoff: RPentHandoff) -> CandidateTool:
        """Generate an unverified candidate tool.

        Args:
            handoff: Descriptor-only L3 failure handoff.

        Returns:
            Candidate descriptor and bound executor.
        """


SynthesisTransport = Callable[[Mapping[str, Any]], Mapping[str, Any]]
CandidateDecoder = Callable[[Mapping[str, Any], RPentHandoff], CandidateTool]


class TransportToolSynthesizer:
    """Adapter for a remote CaP-X transport with an explicit safe decoder.

    The transport returns a descriptor/handle response.  It must not be
    treated as a Python executor.  ``decoder`` is supplied by the sandbox
    integration and is responsible for binding a verified execution handle.
    """

    def __init__(
        self, transport: SynthesisTransport, decoder: CandidateDecoder
    ) -> None:
        """Bind a descriptor transport and trusted executor decoder.

        Args:
            transport: Callable sending a serialized handoff.
            decoder: Callable binding a transport response to an executor.

        Raises:
            TypeError: If either dependency is not callable.
        """
        if not callable(transport) or not callable(decoder):
            raise TypeError("transport and decoder must be callable")
        self._transport = transport
        self._decoder = decoder

    def synthesize(self, handoff: RPentHandoff) -> CandidateTool:
        """Request and decode one candidate tool.

        Args:
            handoff: Descriptor-only L3 failure handoff.

        Returns:
            Decoded candidate tool.

        Raises:
            ToolSynthesisError: If the response is invalid or cannot be decoded.
        """
        response = self._transport(handoff.to_manifest())
        if not isinstance(response, Mapping):
            raise ToolSynthesisError("CaP-X transport must return a mapping")
        if response.get("kind") == "skill":
            raise ToolSynthesisError("CaP-X response must be a tool, not a skill")
        try:
            candidate = self._decoder(response, handoff)
        except Exception as exc:
            raise ToolSynthesisError(f"cannot decode CaP-X candidate: {exc}") from exc
        if not isinstance(candidate, CandidateTool):
            raise ToolSynthesisError("candidate decoder must return CandidateTool")
        return candidate


class MockToolSynthesizer:
    """Deterministic stand-in for CaP-X used for protocol and unit tests."""

    def __init__(self, factories: Mapping[str, ToolExecutor] | None = None) -> None:
        """Configure optional deterministic executors by capability.

        Args:
            factories: Capability-to-executor overrides.
        """
        self._factories = dict(factories or {})

    def synthesize(self, handoff: RPentHandoff) -> CandidateTool:
        """Build a deterministic candidate for the requested capability.

        Args:
            handoff: Descriptor-only L3 failure handoff.

        Returns:
            Candidate using a configured executor or a successful default.
        """
        capability = handoff.required_capability
        executor = self._factories.get(capability)
        if executor is None:

            def executor(
                arguments: Mapping[str, Any], context: Mapping[str, Any]
            ) -> Mapping[str, Any]:
                return {"capability": capability, "solved": True}

        tool_id = f"capx.{capability}"
        spec = ToolSpec(
            tool_id=tool_id,
            name=f"Generated {capability}",
            description=f"Candidate tool synthesized for {capability}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            postconditions=(f"capability:{capability}",),
            constraints=tuple(str(value) for value in handoff.constraints.values()),
            resource_budget=handoff.resource_budget,
            source="capx-mock",
        )
        return CandidateTool(
            spec=spec,
            executor=executor,
            provenance={
                "episode_id": handoff.episode_id,
                "backend": "mock",
                "capability": capability,
            },
        )


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Candidate, its verification report, and registration outcome."""

    candidate: CandidateTool
    verification: VerificationReport
    registered: bool


class ToolSynthesisCoordinator:
    """Orchestrate synthesis, verification, and atomic registry registration."""

    def __init__(
        self, registry: ToolRegistry, verifier: ToolVerifier | None = None
    ) -> None:
        """Bind the target registry and optional tool verifier.

        Args:
            registry: Registry receiving candidates that pass verification.
            verifier: Optional verification policy override.
        """
        self.registry = registry
        self.verifier = verifier or ToolVerifier()

    def synthesize_and_register(
        self,
        handoff: RPentHandoff,
        synthesizer: ToolSynthesizer,
        cases: list[Case] | tuple[Case, ...],
        *,
        precondition: Check | None = None,
        postcondition: Check | None = None,
    ) -> SynthesisResult:
        """Synthesize, verify, and atomically register a candidate.

        Args:
            handoff: Descriptor-only L3 failure handoff.
            synthesizer: Candidate synthesis backend.
            cases: Verification argument and context pairs.
            precondition: Optional verification precondition.
            postcondition: Optional verification postcondition.

        Returns:
            Candidate, report, and whether registration succeeded.
        """
        candidate = synthesizer.synthesize(handoff)
        report = self.verifier.verify(
            candidate.spec,
            candidate.executor,
            cases,
            precondition=precondition,
            postcondition=postcondition,
        )
        if not report.passed:
            return SynthesisResult(candidate, report, False)
        try:
            self.registry.register_verified(candidate.spec, candidate.executor, report)
        except ToolError:
            return SynthesisResult(candidate, report, False)
        return SynthesisResult(candidate, report, True)
