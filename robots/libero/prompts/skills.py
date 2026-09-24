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

"""Optional validated-skill instructions for LIBERO prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rpent.prompt.utils import Numbered, PromptNode

READ_SKILL_LIBRARY = """READ THE VALIDATED SKILL LIBRARY FIRST. Read these four files under
`{{skill_library_dir}}/`:
- `grasp.md`
- `localize.md`
- `transport.md`
- `contact.md`

They contain simulator-validated, cross-task operational experience; every
entry carries its pass rate on debug seeds. Weigh advice by that pass rate, but
do not copy its numeric values: re-derive quantities for the current scene.
Memory and the skill library are separate sources: memory is the agent's own
notes, while the skill library has been validated. In the same audit field that
records the memory files you read, record which skill files you read and which
ones you used. The skill library is read-only for you; only the curator writes
it."""

WRITE_FINDINGS = """WRITE `{{output_dir}}/findings.md` for the later curator after DISTIL and
before `finish`. This requirement overrides any earlier instruction to call
`finish` immediately: do not call `finish` until `findings.md` is written. Do
this whether the cell is solved or unsolved; when DISTIL does not run for an
unsolved cell, this final step still does. Use exactly this format:

## Task: {{suite}} / t{{task}} — <task language from view_env_state>
## Task type: pick-and-place | drawer | stove | microwave | stacking | relative | shelf
## Solved: yes/no (state.libero_terminated), attempts used: <N>

### What kept failing
### What fixed it
### Generalizable patterns (CURATOR READS THIS)
- <pattern>: <做法，用工具名与相对量描述> — proposed skill file: grasp|localize|transport|contact
### Task-specific quirks (not for the library)

Apply all four constraints:
1. List only patterns that may help at least two task types.
2. Do not include absolute coordinates or color/pixel thresholds that apply only
   to this scene.
3. Do not include pass rates. Later validation measures them on independent
   seeds; you neither need nor should report your own.
4. Do not modify the skill library itself. It is read-only for you."""


def add_skill_workflow(
    prompt: PromptNode,
    *,
    include_findings: bool,
) -> PromptNode:
    """Add optional skill-library steps without changing base prompt modules."""
    if not isinstance(prompt, Mapping):
        raise TypeError("LIBERO system prompt must be a mapping")
    workflow = prompt.get("WORKFLOW")
    if not isinstance(workflow, Numbered):
        raise TypeError("LIBERO system prompt WORKFLOW must be numbered")
    steps: tuple[Any, ...] = (READ_SKILL_LIBRARY, *workflow.items)
    if include_findings:
        steps = (*steps, WRITE_FINDINGS)
    return {
        key: Numbered(steps) if key == "WORKFLOW" else value
        for key, value in prompt.items()
    }


__all__ = ["READ_SKILL_LIBRARY", "WRITE_FINDINGS", "add_skill_workflow"]
