"""Error taxonomy for replay: classify observed page state into outcomes.

Matchers are plain dicts so they can live in the artifact JSON and in this
module's defaults. Classification is deterministic: case-insensitive substring
matching, results in matcher order.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutcomeClass(str, Enum):
    BUSINESS_OUTCOME = "business_outcome"
    RECOVERABLE = "recoverable"
    HARD_FAILURE = "hard_failure"


class ErrorMatch(BaseModel):
    """One matched error state."""

    name: str
    outcome_class: OutcomeClass
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="{text_excerpt: str, url: str}",
    )
    recovery: dict[str, Any] | None = None


DEFAULT_MATCHERS: list[dict[str, Any]] = [
    {
        "name": "member_not_found",
        "outcome_class": OutcomeClass.BUSINESS_OUTCOME,
        "text_contains": ["No member found"],
        "url_contains": [],
        "recovery": None,
    },
    {
        "name": "validation_error",
        "outcome_class": OutcomeClass.BUSINESS_OUTCOME,
        "text_contains": ["must be 5 digits"],
        "url_contains": [],
        "recovery": None,
    },
    {
        "name": "session_expired",
        "outcome_class": OutcomeClass.RECOVERABLE,
        "text_contains": ["Session expired"],
        "url_contains": [],
        "recovery": {
            "action": "navigate",
            "params": {"url": "${entry_point}"},
        },
    },
    {
        "name": "permission_denied",
        "outcome_class": OutcomeClass.BUSINESS_OUTCOME,
        "text_contains": ["Permission denied"],
        "url_contains": [],
        "recovery": None,
    },
    {
        "name": "app_error",
        "outcome_class": OutcomeClass.HARD_FAILURE,
        "text_contains": ["Internal Server Error", "Something went wrong"],
        "url_contains": [],
        "recovery": None,
    },
]


def _first_hit(haystack: str, needles: list[str]) -> tuple[int, str] | None:
    """Return (index, needle) of the earliest case-insensitive hit."""
    lowered = haystack.lower()
    best: tuple[int, str] | None = None
    for needle in needles:
        idx = lowered.find(needle.lower())
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, needle)
    return best


def classify(
    text: str,
    url: str,
    matchers: list[dict[str, Any]] | None = None,
) -> list[ErrorMatch]:
    """Classify observed page state against matchers.

    A matcher fires when any of its text_contains substrings appears in the
    text, or any of its url_contains substrings appears in the url (both
    case-insensitive). Returns matches in matcher order. The text excerpt is
    the first 200 characters around the first hit.
    """
    matchers = DEFAULT_MATCHERS if matchers is None else matchers
    text = text or ""
    url = url or ""
    results: list[ErrorMatch] = []
    for m in matchers:
        text_contains = m.get("text_contains") or []
        if isinstance(text_contains, str):
            text_contains = [text_contains]
        url_contains = m.get("url_contains") or []
        if isinstance(url_contains, str):
            url_contains = [url_contains]

        hit = _first_hit(text, list(text_contains))
        excerpt: str
        if hit is not None:
            idx, _needle = hit
            start = max(0, idx - 80)
            excerpt = text[start : idx + 120].strip()
        else:
            url_hit = _first_hit(url, list(url_contains))
            if url_hit is None:
                continue
            excerpt = url[:200]

        results.append(
            ErrorMatch(
                name=m["name"],
                outcome_class=OutcomeClass(m["outcome_class"]),
                evidence={"text_excerpt": excerpt, "url": url},
                recovery=m.get("recovery"),
            )
        )
    return results
