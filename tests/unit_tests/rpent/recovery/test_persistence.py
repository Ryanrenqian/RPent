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

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from rpent.recovery import (
    JsonlWriter,
    read_jsonl,
)
from rpent.recovery.persistence import write_manifest


class TestTrace:
    def test_atomic_manifest_failure_preserves_old_file_and_cleans_temp(self):
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "manifest.json"
            dest.write_text("old")
            with patch(
                "rpent.recovery.persistence.os.replace", side_effect=OSError("failed")
            ):
                with pytest.raises(OSError):
                    write_manifest(dest, {"schema": "test"})
            assert dest.read_text() == "old"
            assert [path.name for path in dest.parent.iterdir()] == ["manifest.json"]
            write_manifest(dest, {"schema": "test"})
            assert json.loads(dest.read_text()) == {"schema": "test"}

    def test_jsonl_writer_is_exclusive_append_only_and_releases_lock(self):
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "trace.jsonl"
            with JsonlWriter(dest) as writer:
                writer.append({"a": 1})
                with pytest.raises(RuntimeError, match="already holds lock"):
                    JsonlWriter(dest)
            assert not Path(f"{dest}.lock").exists()
            with JsonlWriter(dest) as writer:
                writer.append({"a": 2})
            assert read_jsonl(dest) == [{"a": 1}, {"a": 2}]
            with pytest.raises(ValueError):
                writer.append({"a": 3})

    def test_jsonl_writer_lock_is_released_when_process_is_killed(self):
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "trace.jsonl"
            code = "from rpent.recovery import JsonlWriter\nimport sys, time\nwriter = JsonlWriter(sys.argv[1])\nprint('locked', flush=True)\ntime.sleep(300)\n"
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(dest)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
            try:
                assert process.stdout.readline().strip() == "locked"
                with pytest.raises(RuntimeError, match="already holds lock"):
                    JsonlWriter(dest)
                os.kill(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)
                with JsonlWriter(dest) as writer:
                    writer.append({"after_kill": True})
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
            assert read_jsonl(dest) == [{"after_kill": True}]
