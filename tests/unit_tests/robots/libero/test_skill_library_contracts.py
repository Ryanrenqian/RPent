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

import argparse
import hashlib
import re
from pathlib import Path

import pytest

from robots.libero.robot_spec import get_robot_spec
from rpent.memory import MemoryManager

SKILL_FILES = ("grasp.md", "localize.md", "transport.md", "contact.md")
SEED_DIR = Path(__file__).parents[4] / "robots" / "libero" / "skill_library_seed"

# Generated directly from 573565f with PromptBundle.render using BASE_VARIABLES.
BASELINE_SHA256 = {
    (
        "eval",
        "system",
    ): "5d235ea6a4dc6a949db5779e726ec2c6fa57a3e63191a82195e6168c3132c858",
    (
        "eval",
        "user",
    ): "dc50b6b135f2fac74b146e084078d59848d7b1a047aeb767323135b5fdc97abb",
    (
        "explore",
        "system",
    ): "4fdc1265d7125dffc788ddd432f8dc2e9f045e1ca72a7b64d633646fe145b051",
    (
        "explore",
        "user",
    ): "dc50b6b135f2fac74b146e084078d59848d7b1a047aeb767323135b5fdc97abb",
}
BASE_VARIABLES = {
    "suite": "libero_object_task",
    "task": 2,
    "seed": 0,
    "recipe_tag": "object_task_t2_s0",
    "memory_dir": "/baseline/memory/libero",
    "reference_tag": "object_task_t2_s0",
    "memory_inbox": ("/baseline/memory/libero/_internal/inbox/object_task_t2_s0"),
    "session_number": 1,
    "session_max": 3,
    "output_dir": "/baseline/output",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--memory-profile", choices=["hf", "local"], default=None)
    parser.add_argument("--memory-dir", default=None)
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    return parser


def _write_skill_library(path: Path) -> None:
    path.mkdir()
    for name in SKILL_FILES:
        (path / name).write_text(f"# {name}\n")


@pytest.mark.parametrize("mode", ["eval", "explore"])
@pytest.mark.parametrize("variant", ["system", "user"])
def test_default_prompts_match_573565f_byte_for_byte(
    mode: str,
    variant: str,
) -> None:
    variables = {
        **BASE_VARIABLES,
        "mode": mode,
        "memory_profile": "local" if mode == "explore" else "hf",
    }

    rendered = get_robot_spec().prompts.render(variant, variables=variables)

    assert (
        hashlib.sha256(rendered.encode()).hexdigest() == BASELINE_SHA256[mode, variant]
    )


def test_skill_library_cli_resolves_and_validates_required_files(
    tmp_path: Path,
) -> None:
    skill_library = tmp_path / "skills"
    _write_skill_library(skill_library)
    args = _parser().parse_args(
        [
            "--suite",
            "libero_goal",
            "--task",
            "0",
            "--skill-library-dir",
            str(skill_library),
        ]
    )

    config = get_robot_spec().parse_config(args)

    assert config.prompt_vars["skill_library_dir"] == str(skill_library.resolve())

    (skill_library / "contact.md").unlink()
    with pytest.raises(ValueError, match="contact[.]md"):
        get_robot_spec().parse_config(args)

    args.skill_library_dir = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="directory not found"):
        get_robot_spec().parse_config(args)


def test_skill_library_cli_defaults_to_disabled() -> None:
    args = _parser().parse_args(["--suite", "libero_goal", "--task", "0"])

    config = get_robot_spec().parse_config(args)

    assert args.skill_library_dir is None
    assert "skill_library_dir" not in config.prompt_vars


def test_skill_library_must_be_separate_from_memory(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    skill_library = memory_dir / "skills"
    skill_library.mkdir(parents=True)
    for name in SKILL_FILES:
        (skill_library / name).write_text("")
    args = _parser().parse_args(
        [
            "--suite",
            "libero_goal",
            "--task",
            "0",
            "--memory-dir",
            str(memory_dir),
            "--memory-profile",
            "local",
            "--skill-library-dir",
            str(skill_library),
        ]
    )

    with pytest.raises(ValueError, match="must be separate"):
        get_robot_spec().parse_config(args)


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_skill_library_prompt_has_mode_specific_steps_and_no_placeholders(
    mode: str,
) -> None:
    variables = {
        **BASE_VARIABLES,
        "mode": mode,
        "memory_profile": "local" if mode == "explore" else "hf",
        "skill_library_dir": "/validated/skills",
    }

    system = get_robot_spec().prompts.render("system", variables=variables)
    user = get_robot_spec().prompts.render("user", variables=variables)

    assert "{{" not in system
    assert "{{" not in user
    assert system.index("READ THE VALIDATED SKILL LIBRARY FIRST") < system.index(
        "READ MEMORY FIRST"
    )
    assert "grasp.md" in system
    assert "debug seeds" in system
    assert "which skill files you read" in system
    assert "which\n   ones you used" in system
    assert ("findings.md" in system) is (mode == "explore")
    if mode == "explore":
        assert system.index("DISTIL — consolidate this cell into memory") < (
            system.index("WRITE `/baseline/output/findings.md`")
        )
        assert "### Generalizable patterns (CURATOR READS THIS)" in system
        assert "Do not include pass rates" in system


@pytest.mark.parametrize("memory_access", ["read_only", "inbox_write"])
def test_skill_library_is_readable_and_not_writable_in_both_modes(
    tmp_path: Path,
    memory_access: str,
) -> None:
    skill_library = tmp_path / "skills"
    _write_skill_library(skill_library)
    manager = MemoryManager(
        tmp_path / "memory",
        memory_access=memory_access,
        inbox_cell_tag="cell" if memory_access == "inbox_write" else None,
        skill_library_dir=skill_library,
    )
    bindings = manager.get_common_tool_bindings()
    read = bindings["read_text_file"][1]
    write = bindings["write_text_file"][1]
    list_dir = bindings["list_dir"][1]

    assert read(str(skill_library / "grasp.md"))["content"] == "# grasp.md\n"
    assert list_dir(str(skill_library))["files"] == sorted(SKILL_FILES)
    with pytest.raises(
        PermissionError,
        match="skill library is read-only for the acting agent; the curator writes it",
    ):
        write(str(skill_library / "grasp.md"), "changed")

    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    escape = skill_library / "escape.md"
    escape.symlink_to(outside)
    with pytest.raises(
        PermissionError,
        match="skill library is read-only for the acting agent; the curator writes it",
    ):
        write(str(escape), "changed")
    assert outside.read_text() == "outside"
    alias = tmp_path / "skill-alias.md"
    alias.symlink_to(skill_library / "grasp.md")
    with pytest.raises(
        PermissionError,
        match="skill library is read-only for the acting agent; the curator writes it",
    ):
        write(str(alias), "changed")
    assert (skill_library / "grasp.md").read_text() == "# grasp.md\n"


def test_default_manager_preserves_all_three_file_tool_boundaries(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    published = memory_dir / "global" / "published.md"
    published.parent.mkdir(parents=True)
    published.write_text("published")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "note.md"
    outside.write_text("outside")
    bindings = MemoryManager(memory_dir).get_common_tool_bindings()
    read = bindings["read_text_file"][1]
    write = bindings["write_text_file"][1]
    list_dir = bindings["list_dir"][1]

    assert read(str(published))["content"] == "published"
    assert list_dir(str(published.parent))["files"] == ["published.md"]
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        write(str(published), "changed")
    assert read(str(outside))["content"] == "outside"
    assert list_dir(str(outside_dir))["files"] == ["note.md"]
    assert write(str(outside), "changed")["bytes_written"] == 7
    assert outside.read_text() == "changed"


def test_seed_skill_files_are_empty_structural_skeletons() -> None:
    assert {path.name for path in SEED_DIR.glob("*.md")} >= {*SKILL_FILES}
    for name in SKILL_FILES:
        text = (SEED_DIR / name).read_text()
        table_lines = [line for line in text.splitlines() if line.startswith("|")]

        assert table_lines == [
            "| pattern | when to apply | how | debug-seed pass rate | source task |",
            "| --- | --- | --- | --- | --- |",
        ]
        assert re.search(r"\d+\.\d+", text) is None
