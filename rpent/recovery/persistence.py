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

"""Versioned JSON manifests for tools and experiential skills.

Manifests intentionally persist descriptors and evidence, never Python
executors or generated source. Executors must be re-bound explicitly by the
runtime after loading a manifest.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

TOOL_MANIFEST_SCHEMA = "agentic-em/tool-manifest/v1"
SKILL_MANIFEST_SCHEMA = "agentic-em/skill-manifest/v1"


class ManifestError(ValueError):
    """Raised for malformed or incompatible manifests."""


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON manifest.

    Args:
        path: Destination path.
        payload: JSON-serializable manifest mapping.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class JsonlWriter:
    """Append-only JSONL writer with an exclusive advisory file lock."""

    def __init__(self, path: str | Path) -> None:
        """Open and exclusively lock an append-only JSONL stream.

        Args:
            path: JSONL destination path.

        Raises:
            RuntimeError: If another writer already holds the advisory lock.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            raise RuntimeError(f"JSONL writer already holds lock: {self.path}") from exc

    def append(self, payload: Mapping[str, Any]) -> None:
        """Append and flush one JSON object.

        Args:
            payload: JSON-serializable record mapping.

        Raises:
            ValueError: If this writer is closed.
        """
        if self._stream.closed:
            raise ValueError("JSONL writer is closed")
        line = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        self._stream.write(line)
        self._stream.flush()

    def close(self) -> None:
        """Release the advisory lock and close the stream."""
        if not self._stream.closed:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()

    def __enter__(self) -> JsonlWriter:
        """Return this open writer for a context manager."""
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        """Close the writer when its context exits."""
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read mapping records from a JSONL file.

    Args:
        path: JSONL source path.

    Returns:
        Parsed records in file order.

    Raises:
        ValueError: If a non-mapping JSON value is present.
    """
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("JSONL records must be objects")
    return records


def read_manifest(path: str | Path, expected_schema: str) -> dict[str, Any]:
    """Read and validate a schema-tagged JSON manifest.

    Args:
        path: Manifest source path.
        expected_schema: Required schema identifier.

    Returns:
        Parsed manifest mapping.

    Raises:
        ManifestError: If the file is invalid, not a mapping, or has the wrong
            schema identifier.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ManifestError(f"manifest {path} is not {expected_schema}")
    return payload
