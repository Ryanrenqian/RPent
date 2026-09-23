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

"""Shared classification of failure-reporting toolkit results."""

from __future__ import annotations

from typing import Any, Mapping


def classify_tool_result_failure(
    raw_result: Mapping[str, Any],
) -> tuple[str | None, Any | None]:
    """Return the reported failure source and error value for a tool result.

    The action result may be nested under ``log.result`` by environment state
    capture. Truthy errors take precedence over explicit ``success=False``.

    Args:
        raw_result: Final mapping returned by toolkit execution.

    Returns:
        A ``(source, error_value)`` pair. Both values are ``None`` for success;
        ``error_value`` is also ``None`` when failure is reported only through
        ``success=False``.
    """
    action_result = raw_result
    log = raw_result.get("log")
    if isinstance(log, Mapping) and isinstance(log.get("result"), Mapping):
        action_result = log["result"]

    error_value = raw_result.get("error") or action_result.get("error")
    if error_value:
        return "result_error", error_value
    if raw_result.get("success") is False or action_result.get("success") is False:
        return "result_success_false", None
    return None, None
