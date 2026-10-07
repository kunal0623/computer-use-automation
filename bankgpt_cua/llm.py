"""LLM client layer for the BankGPT computer-use agent.

Defines the decision schema exchanged between the agent loop and the model,
an OpenAI-compatible HTTP client, and a scripted mock client for
deterministic pipeline testing. No em dashes are used anywhere in this file.

OpenAI-compatible client configuration (environment variables):
  LLM_API_KEY      API key (Bearer token). Falls back to OPENAI_API_KEY.
  LLM_BASE_URL     Base URL of the chat-completions endpoint. Falls back to
                   OPENAI_BASE_URL. Defaults to the Google AI Studio
                   (Gemini) OpenAI-compatible endpoint:
                   https://generativelanguage.googleapis.com/v1beta/openai/
  LLM_MODEL        Model name. Defaults to gemini-3.8-flash.

The default setup targets a Google AI Studio (Gemini) key used through its
OpenAI-compatible endpoint: POST {base_url}/chat/completions with an
Authorization: Bearer <key> header. The client itself is endpoint-agnostic,
so any OpenAI-compatible backend works by overriding the env vars above.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field


class ActionDecision(BaseModel):
    """A single action chosen by the LLM for the current page state."""

    action: Literal["navigate", "click", "fill", "press", "select", "wait", "read", "done", "escalate"]
    target: dict[str, Any] | None = None
    value: str | None = None
    key: str | None = None
    ms: int | None = None
    reasoning: str = ""
    summary: str | None = None
    escalate_reason: str | None = None


class ActionRequest(BaseModel):
    """The observation payload sent to the LLM to pick the next action."""

    goal: str
    step_index: int
    history: list[dict[str, Any]]
    observation: dict[str, Any] = Field(
        description="Keys: url, title, a11y, dom_excerpt",
    )


class LLMClient(Protocol):
    """Protocol every LLM backend must satisfy."""

    def decide(self, req: ActionRequest) -> ActionDecision: ...


_ACTION_SCHEMA_JSON = json.dumps(
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "click", "fill", "press", "select", "wait", "read", "done", "escalate"],
            },
            "target": {
                "type": ["object", "null"],
                "description": "Target hints, e.g. {'text': 'Log in'} or {'role': 'button', 'name': 'Search member'} or {'css': '#member-id'}",
            },
            "value": {"type": ["string", "null"], "description": "Value for fill/select, or URL for navigate"},
            "key": {"type": ["string", "null"], "description": "Key for press, e.g. 'Enter'"},
            "ms": {"type": ["integer", "null"], "description": "Milliseconds for wait"},
            "reasoning": {"type": "string", "description": "One or two sentences on why this action is the right next step"},
            "summary": {"type": ["string", "null"], "description": "Final summary when action is 'done'"},
            "escalate_reason": {"type": ["string", "null"], "description": "Why escalation is needed when action is 'escalate'"},
        },
        "required": ["action", "reasoning"],
    },
    indent=2,
)

_SYSTEM_PROMPT = f"""You are a computer-use agent operating a banking web application.

Your task: choose the next single action toward the stated goal, based on the
current page observation (URL, title, accessibility tree, DOM excerpt) and the
action history.

Rules:
1. Output ONLY valid JSON matching this schema, with no surrounding text,
   markdown fences, or commentary:
{_ACTION_SCHEMA_JSON}
2. Pick exactly one action per response. Prefer the smallest safe step.
3. Target elements by stable, human-meaningful hints. Prefer role+name,
   then label, then placeholder, then CSS selector, then XPath, then visible
   text, in that order of robustness.
4. Never invent page state that is not visible in the observation.
5. Use action "read" to extract visible text (balances, names, statuses) when
   the goal requires reading a value.
6. Use action "done" only when the goal is fully achieved, and include a
   summary of the outcome, including any values the goal asked for.
7. Use action "escalate" when you are blocked, when the page asks for
   credentials or sensitive confirmation outside the goal, or when the action
   would be destructive or irreversible. Explain why in escalate_reason.
8. Do not fabricate credentials. If a login page appears and no credentials
   were provided in the goal context, escalate.
"""


class OpenAICompatClient:
    """OpenAI-compatible LLM client (default backend: Gemini via Google AI Studio).

    Reads configuration from the environment (see module docstring):
      LLM_API_KEY (fallback OPENAI_API_KEY), required
      LLM_BASE_URL (fallback OPENAI_BASE_URL), default
        https://generativelanguage.googleapis.com/v1beta/openai/
      LLM_MODEL, default gemini-3.8-flash

    Authenticates with a Bearer token in the Authorization header and POSTs
    to {base_url}/chat/completions. The API key is never logged.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        key = (
            api_key
            if api_key is not None
            else os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "OpenAICompatClient requires an API key: set the LLM_API_KEY "
                "environment variable (or OPENAI_API_KEY as a fallback), "
                "or pass api_key explicitly."
            )
        self._api_key = key
        self._base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        ).rstrip("/")
        self._model = model or os.environ.get("LLM_MODEL", "gemini-3.8-flash")
        self._timeout_s = timeout_s

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _extract_json(content: Any) -> Any:
        """Parse model content into a Python object, tolerating code fences.

        Some OpenAI-compatible backends wrap JSON in markdown fences despite
        the JSON-only instruction; strip them before parsing.
        """
        if not isinstance(content, str):
            return content
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    def decide(self, req: ActionRequest) -> ActionDecision:
        user_payload = {
            "goal": req.goal,
            "step_index": req.step_index,
            "history": req.history,
            "observation": req.observation,
        }
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # The sandbox exports proxy env vars whose no_proxy list contains
        # bare IPv6 literals that httpx cannot parse as URL patterns, so
        # trust_env must stay off. We still honor an explicit HTTPS proxy
        # for egress instead of bypassing it.
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        verify = os.environ.get("SSL_CERT_FILE", True)
        # Retry transient overload / rate-limit responses with backoff.
        for attempt in range(4):
            with httpx.Client(
                timeout=self._timeout_s, trust_env=False, proxy=proxy, verify=verify
            ) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions", json=body, headers=headers
                )
            if getattr(resp, "status_code", 200) in (429, 503) and attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LLM request failed with status {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError("LLM response was not valid JSON") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response missing choices[0].message.content") from exc
        try:
            parsed = self._extract_json(content)
        except ValueError as exc:
            raise RuntimeError(f"LLM content was not valid JSON: {str(content)[:500]!r}") from exc
        try:
            return ActionDecision.model_validate(parsed)
        except Exception as exc:
            raise RuntimeError(f"LLM output did not match ActionDecision schema: {exc}") from exc


class ScriptedMockClient:
    """Deterministic stand-in for the LLM: replays a fixed decision script.

    Used for deterministic pipeline testing. This is NOT a real LLM run.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = [ActionDecision.model_validate(step) for step in script]
        self._cursor = 0

    def decide(self, req: ActionRequest) -> ActionDecision:
        if self._cursor < len(self._script):
            decision = self._script[self._cursor]
            self._cursor += 1
            return decision
        return ActionDecision(
            action="done",
            reasoning="Script exhausted, no further scripted steps remain.",
            summary="Script exhausted",
        )


def default_demo_script(entry_url: str) -> list[dict[str, Any]]:
    """Scripted decision sequence for 'Look up member 12345 and read their current savings balance'."""
    return [
        {
            "action": "navigate",
            "target": None,
            "value": entry_url,
            "reasoning": "Start by navigating to the mock bank entry point.",
        },
        {
            "action": "fill",
            "target": {"text": "Username"},
            "value": "operator",
            "reasoning": "Fill the Username field on the login form.",
        },
        {
            "action": "fill",
            "target": {"text": "Password"},
            "value": "operator",
            "reasoning": "Fill the Password field on the login form.",
        },
        {
            "action": "click",
            "target": {"text": "Log in"},
            "reasoning": "Submit the login form.",
        },
        {
            "action": "fill",
            "target": {"text": "Member ID"},
            "value": "12345",
            "reasoning": "Fill the Member ID field on the search form.",
        },
        {
            "action": "click",
            "target": {"text": "Search member"},
            "reasoning": "Submit the member search.",
        },
        {
            "action": "read",
            "target": {"text": "Savings balance"},
            "reasoning": "Read the member's savings balance from the member page.",
        },
        {
            "action": "done",
            "target": None,
            "reasoning": "Goal achieved, the balance was read from the member page.",
            "summary": "Member 12345 current savings balance: $4,321.09",
        },
    ]
