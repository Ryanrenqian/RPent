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

"""Failure-to-experience runtime primitives.

This package intentionally keeps the first AgenticEM control-plane contracts
independent from any simulator or remote model service.  A *tool* is an
executable state-transition operator; an experiential *skill* is a reusable
problem-solving playbook that orchestrates tools.
"""

from .analysis import (
    diagnoser_confusion_matrix,
    experience_gain,
    router_confusion_matrix,
    skill_reuse_rate,
)
from .budget import BudgetLedger, BudgetLimits
from .classify import (
    CellRecord,
    CellSignals,
    CellVerdict,
    InfraProtocolClassifier,
    signals_from_cell_dir,
)
from .diagnose import DiagnosisResult, DiagnosisSignals, FailureDiagnoser
from .events import (
    ExecutionEvent,
    FailureEvent,
    RecoveryAction,
    RecoveryDecision,
    RecoveryLevel,
    ToolGapEvent,
)
from .handoff import HANDOFF_SCHEMA, RPentHandoff
from .libero_evidence import map_libero_evidence
from .persistence import JsonlWriter, read_jsonl
from .router import FailureRouter
from .runtime import RuntimeResult, SkillRuntime
from .skills import (
    ExperienceSkillBank,
    SkillPlaybook,
    SkillStep,
    SkillValidation,
)
from .synthesis import (
    CandidateTool,
    MockToolSynthesizer,
    SynthesisResult,
    ToolSynthesisCoordinator,
    ToolSynthesizer,
    TransportToolSynthesizer,
)
from .tool_gap import ToolGapAdapter
from .toolkit_bridge import ToolkitBackedRegistry
from .tools import (
    ToolCall,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from .verification import (
    OutcomeVerifier,
    SkillVerifier,
    SnapshotToolVerifier,
    ToolVerifier,
    VerificationReport,
)

__all__ = [
    "ExecutionEvent",
    "FailureEvent",
    "RecoveryDecision",
    "RecoveryLevel",
    "RecoveryAction",
    "ToolGapEvent",
    "BudgetLedger",
    "BudgetLimits",
    "DiagnosisResult",
    "DiagnosisSignals",
    "FailureDiagnoser",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ExperienceSkillBank",
    "SkillPlaybook",
    "SkillStep",
    "SkillValidation",
    "FailureRouter",
    "RuntimeResult",
    "SkillRuntime",
    "OutcomeVerifier",
    "SnapshotToolVerifier",
    "SkillVerifier",
    "ToolVerifier",
    "VerificationReport",
    "HANDOFF_SCHEMA",
    "RPentHandoff",
    "CandidateTool",
    "MockToolSynthesizer",
    "SynthesisResult",
    "ToolSynthesisCoordinator",
    "ToolSynthesizer",
    "TransportToolSynthesizer",
    "ToolGapAdapter",
    "ToolkitBackedRegistry",
    "map_libero_evidence",
    "CellRecord",
    "CellSignals",
    "CellVerdict",
    "InfraProtocolClassifier",
    "signals_from_cell_dir",
    "JsonlWriter",
    "read_jsonl",
    "router_confusion_matrix",
    "diagnoser_confusion_matrix",
    "experience_gain",
    "skill_reuse_rate",
]
