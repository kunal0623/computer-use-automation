"""Shared discovery helpers: run a goal, build a capability artifact.

This module holds the pieces that both the ``tools/discover.py`` CLI and
the Studio chat page use: the default input/output/error-policy specs for
the mock-bank member-lookup flow, the literal-to-placeholder
canonicalization step, and the artifact build-and-save routine. One source
of truth keeps the CLI and the UI from drifting apart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from bankgpt_cua.artifact import ArtifactBuilder, CapabilityArtifact

DEFAULT_INPUTS = [
    {
        "name": "member_id",
        "type": "string",
        "required": True,
        "description": "Five-digit member ID to look up.",
        "pattern": "^[0-9]{5}$",
        "example": "12345",
        "redact_in_logs": True,
    }
]

DEFAULT_OUTPUTS = [
    {
        "name": "savings_balance",
        "type": "string",
        "description": "Current savings account balance as a decimal string, e.g. 4321.09.",
        "extraction": {
            "step_index": 6,
            "strategy": {"type": "text", "value": "Savings balance"},
            "postprocess": "extract_currency",
        },
    }
]

DEFAULT_ERROR_POLICY = [
    {
        "name": "member_not_found",
        "outcome_class": "business_outcome",
        "match": {"text_contains": ["No member found"]},
        "outcome": {
            "code": "MEMBER_NOT_FOUND",
            "message_template": "No member found for ID {member_id}.",
        },
        "recovery": None,
    },
    {
        "name": "validation_error",
        "outcome_class": "business_outcome",
        "match": {"text_contains": ["must be 5 digits"]},
        "outcome": {
            "code": "INVALID_INPUT",
            "message_template": "The member ID was rejected by the application: invalid format.",
        },
        "recovery": None,
    },
    {
        "name": "session_expired",
        "outcome_class": "recoverable",
        "match": {"text_contains": ["Session expired"]},
        "outcome": {"code": "SESSION_EXPIRED", "message_template": "Session expired."},
        "recovery": {"action": "navigate", "params": {"url": "${entry_point}"}},
    },
    {
        "name": "permission_denied",
        "outcome_class": "business_outcome",
        "match": {"text_contains": ["Permission denied"]},
        "outcome": {
            "code": "PERMISSION_DENIED",
            "message_template": "The operator account is not permitted to view this page.",
        },
        "recovery": None,
    },
]


def parameterize(artifact: CapabilityArtifact, inputs: list[dict[str, Any]]) -> None:
    """Replace recorded literal values with ${input} placeholders.

    The discovery run uses concrete values (e.g. member ID 12345). For the
    saved capability to be reusable, any step param or checkpoint that
    exactly matches a declared input's example is rewritten to reference
    the input name. This is a deliberate, minimal canonicalization step:
    replay substitutes the caller's inputs before executing.
    """
    for spec in inputs:
        name = spec.get("name")
        example = spec.get("example")
        if not name or example is None:
            continue
        placeholder = "${" + name + "}"
        example_str = str(example)
        for step in artifact.steps:
            params = step.params or {}
            for key, value in list(params.items()):
                if isinstance(value, str) and value == example_str:
                    params[key] = placeholder
                elif isinstance(value, str) and example_str in value:
                    params[key] = value.replace(example_str, placeholder)
        cond = artifact.success_condition
        if cond and isinstance(cond.value, str) and example_str in cond.value:
            cond.value = cond.value.replace(example_str, placeholder)
        if isinstance(artifact.description, str) and example_str in artifact.description:
            artifact.description = artifact.description.replace(example_str, placeholder)


def default_secret_values(inputs: list[dict[str, Any]]) -> list[str]:
    """Example values flagged redact_in_logs, for masking in run logs."""
    return [
        str(spec["example"])
        for spec in inputs
        if spec.get("redact_in_logs") and spec.get("example") is not None
    ]


def build_artifact(
    run: Any,
    *,
    artifact_id: str,
    name: str,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    error_policy: list[dict[str, Any]] | None = None,
    review_notes: str = "",
) -> CapabilityArtifact:
    """Build a parameterized capability artifact from a completed agent run."""
    artifact = ArtifactBuilder.from_run(
        run,
        artifact_id=artifact_id,
        name=name,
        inputs=inputs if inputs is not None else DEFAULT_INPUTS,
        outputs=outputs if outputs is not None else DEFAULT_OUTPUTS,
        error_policy=error_policy if error_policy is not None else DEFAULT_ERROR_POLICY,
        review_notes=review_notes,
    )
    parameterize(artifact, inputs if inputs is not None else DEFAULT_INPUTS)
    return artifact


def save_artifact_json(artifact: CapabilityArtifact, run_dir: str | Path) -> Path:
    """Write artifact.json into a run directory and return its path."""
    payload = artifact.model_dump()
    path = Path(run_dir) / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def redact_summary_line(line: str, secret_values: list[str]) -> str:
    """Mask secret example values in a one-line run summary."""
    for secret in secret_values:
        line = line.replace(secret, "***")
    return line
