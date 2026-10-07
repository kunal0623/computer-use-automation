"""Guardrails for the BankGPT computer-use agent.

This module provides three things:

1. ``Policy``: an allowlist policy for routes and action types, plus a
   risky-action classification that either blocks or requires human
   confirmation.
2. ``check_action``: evaluates one action dict against a ``Policy`` and the
   current page URL, returning {"allowed", "requires_human", "reason"}.
3. Redaction helpers (``redact``, ``redact_dict``) for scrubbing secrets
   from logs and evidence.

Limits, stated honestly:

- URL and action allowlists are regex/allowlist based. They only cover
  what is configured; a permissive pattern (for example allowing all of
  localhost) still trusts everything served on those ports.
- Redaction is heuristic regex matching. It catches common formats for
  SSNs, 16-digit card numbers, bearer tokens, and "password: ..." style
  assignments, but it will miss exotic formats and can over-redact
  innocent strings that happen to look like secrets. Literal secrets
  passed via ``extra_values`` are always redacted exactly.
- Risky-action classification for clicks is based on target text matching
  (for example "click:Close savings account"). It does not understand
  page semantics: a destructive control with different wording will not
  be flagged unless the action carries params {"irreversible": True},
  which is the explicit signal agent code should set for state-changing
  operations.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class Policy(BaseModel):
    """Allowlist and risky-action policy for one run."""

    allowed_route_patterns: list[str] = Field(
        default_factory=lambda: [r"^http://127\.0\.0\.1:\d+/.*", r"^http://localhost:\d+/.*"]
    )
    allowed_action_types: list[str] = Field(
        default_factory=lambda: ["navigate", "click", "fill", "press", "select", "wait", "read"]
    )
    risky_actions: list[str] = Field(
        default_factory=lambda: ["click:Close savings account"]
    )
    risky_mode: Literal["block", "confirm"] = "block"
    blocked_url_patterns: list[str] = Field(default_factory=list)

    def check_action(self, action: dict, url: str) -> dict:
        """Evaluate one action against this policy.

        Returns {"allowed": bool, "requires_human": bool, "reason": str}.
        This method form exists so agent and replay code can treat the
        policy object as the guardrail entry point.
        """
        return check_action(action, url, self)


def _target_text(target: Any) -> str:
    """Best-effort concatenation of the human-readable target text."""
    if not isinstance(target, dict):
        return ""
    parts: list[str] = []
    for key in ("text", "label", "name", "value", "aria_label"):
        value = target.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return " ".join(parts)


def _risky_match(action: dict, policy: Policy) -> str | None:
    """Return a description of the matched risky rule, or None.

    A rule is either "action_type:substring" (substring matched
    case-insensitively against the target text) or a bare action type.
    Any action carrying params {"irreversible": True} is always risky.
    """
    action_type = str(action.get("action", ""))
    params = action.get("params") or {}
    if isinstance(params, dict) and params.get("irreversible") is True:
        return f"{action_type} (params mark it irreversible)"
    text = _target_text(action.get("target")).lower()
    for rule in policy.risky_actions:
        if ":" in rule:
            rule_type, _, needle = rule.partition(":")
            if action_type == rule_type.strip() and needle.strip().lower() in text:
                return rule
        elif action_type == rule.strip():
            return rule
    return None


def check_action(action: dict, url: str, policy: Policy | None = None) -> dict:
    """Evaluate one action dict against the policy.

    Returns {"allowed": bool, "requires_human": bool, "reason": str}.
    ``policy`` defaults to a stock ``Policy`` so the two-argument contract
    ``check_action(action, url)`` keeps working; callers with a configured
    policy pass it explicitly.
    """
    if policy is None:
        policy = Policy()

    action_type = str(action.get("action", ""))

    if action_type == "navigate":
        destination = str(action.get("value") or "")
        for pattern in policy.blocked_url_patterns:
            if re.search(pattern, destination):
                return {
                    "allowed": False,
                    "requires_human": False,
                    "reason": f"URL matches blocked pattern: {destination}",
                }
        if not any(re.search(p, destination) for p in policy.allowed_route_patterns):
            return {
                "allowed": False,
                "requires_human": False,
                "reason": f"URL outside allowlist: {destination}",
            }

    if action_type not in policy.allowed_action_types:
        return {
            "allowed": False,
            "requires_human": False,
            "reason": f"Action type '{action_type}' is not in the allowlist",
        }

    risky = _risky_match(action, policy)
    if risky is not None:
        if policy.risky_mode == "block":
            return {
                "allowed": False,
                "requires_human": False,
                "reason": f"Risky action blocked by policy: {risky}",
            }
        return {
            "allowed": True,
            "requires_human": True,
            "reason": f"Risky action requires human confirmation: {risky}",
        }

    return {"allowed": True, "requires_human": False, "reason": "ok"}


# Name and compiled regex pairs. Order matters: apply SSN before card so a
# hyphenated SSN is not partially eaten by a looser pattern.
REDACTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("card", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("bearer", re.compile(r"[Bb]earer\s+[A-Za-z0-9\-._~+/]+=*")),
    ("password", re.compile(r"(password\s*[:=]\s*)\S+", re.IGNORECASE)),
]


def redact(text: str, extra_values: list[str] | None = None) -> str:
    """Redact secrets in ``text``.

    Regex matches are replaced with "[REDACTED:<name>]". Literal values in
    ``extra_values`` (for example input parameter values flagged with
    redact_in_logs) are replaced with "[REDACTED:param]".
    """
    if not isinstance(text, str):
        return text
    redacted = text
    for name, pattern in REDACTION_RULES:
        if name == "password":
            redacted = pattern.sub(lambda m: m.group(1) + "[REDACTED:password]", redacted)
        else:
            redacted = pattern.sub(f"[REDACTED:{name}]", redacted)
    for secret in extra_values or []:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED:param]")
    return redacted


def redact_dict(data: Any, extra_values: list[str] | None = None) -> Any:
    """Recursively redact string values inside dicts, lists, and tuples."""
    if isinstance(data, dict):
        return {key: redact_dict(value, extra_values) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        redacted = [redact_dict(item, extra_values) for item in data]
        return type(data)(redacted) if isinstance(data, tuple) else redacted
    if isinstance(data, str):
        return redact(data, extra_values)
    return data
