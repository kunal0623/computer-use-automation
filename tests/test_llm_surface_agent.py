"""Tests for bankgpt_cua.llm, bankgpt_cua.surface, and bankgpt_cua.agent."""

import os
import urllib.parse

import pytest
from playwright.sync_api import sync_playwright

from bankgpt_cua.agent import AgentRun, run_goal
from bankgpt_cua.llm import (
    ActionDecision,
    ActionRequest,
    OpenAICompatClient,
    ScriptedMockClient,
    default_demo_script,
)
from bankgpt_cua.surface import (
    SurfaceState,
    TargetNotFoundError,
    WebSurface,
    build_strategy_chain,
    resolve_strategies,
)

MOCK_HTML = """<!DOCTYPE html>
<html><head><title>Mock Bank</title></head><body>
<h1>Mock Bank</h1>
<form id="login">
  <table><tr><td>
    <table><tr><td>
      <label for="u">Username</label>
      <input id="u" name="username" type="text">
    </td></tr></table>
  </td></tr><tr><td>
    <label for="p">Password</label>
    <input id="p" name="password" type="password">
  </td></tr></table>
  <button type="button">Log in</button>
</form>
<form id="search">
  <label>Member ID <input id="m" name="member_id" type="text"></label>
  <button type="button">Search member</button>
</form>
<table id="outer"><tr><td>
  <table id="member"><tr><td>Savings balance</td><td>$4,321.09</td></tr></table>
</td></tr></table>
</body></html>"""


def data_url(html: str = MOCK_HTML) -> str:
    return "data:text/html;charset=utf-8," + urllib.parse.quote(html, safe="")


@pytest.fixture()
def surface():
    s = WebSurface()
    s.start(data_url())
    yield s
    s.close()


# ---------------------------------------------------------------- llm.py ---


def test_scripted_mock_client_sequencing():
    script = [
        {"action": "click", "target": {"text": "Log in"}, "reasoning": "submit"},
        {"action": "done", "reasoning": "finished", "summary": "ok"},
    ]
    client = ScriptedMockClient(script)
    req = ActionRequest(
        goal="g", step_index=0, history=[],
        observation={"url": "u", "title": "t", "a11y": "a", "dom_excerpt": "d"},
    )
    first = client.decide(req)
    second = client.decide(req)
    third = client.decide(req)
    assert first.action == "click"
    assert first.target == {"text": "Log in"}
    assert second.action == "done"
    assert third.action == "done"
    assert third.summary == "Script exhausted"


def test_default_demo_script_structure():
    script = default_demo_script("https://bank.example/login")
    decisions = [ActionDecision.model_validate(d) for d in script]
    actions = [d.action for d in decisions]
    assert actions == ["navigate", "fill", "fill", "click", "fill", "click", "read", "done"]
    assert decisions[0].value == "https://bank.example/login"
    assert decisions[1].target == {"text": "Username"}
    assert decisions[1].value == "operator"
    assert decisions[2].target == {"text": "Password"}
    assert decisions[2].value == "operator"
    assert decisions[3].target == {"text": "Log in"}
    assert decisions[4].target == {"text": "Member ID"}
    assert decisions[4].value == "12345"
    assert decisions[5].target == {"text": "Search member"}
    assert decisions[6].target == {"text": "Savings balance"}
    assert all(d.reasoning for d in decisions)
    assert "$4,321.09" in (decisions[-1].summary or "")


def test_openai_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        OpenAICompatClient()


def test_openai_compat_client_env_defaults(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = OpenAICompatClient()
    assert client._base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert client.model == "gemini-3.8-flash"


def test_openai_compat_client_env_fallbacks(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = OpenAICompatClient()
    assert client._api_key == "legacy-key"
    assert client._base_url == "https://custom.example/v1"

    monkeypatch.setenv("LLM_API_KEY", "primary-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://primary.example/v1/")
    client = OpenAICompatClient()
    assert client._api_key == "primary-key"
    assert client._base_url == "https://primary.example/v1"


class _FakeHTTPResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeHTTPClient:
    captured = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None, headers=None):
        type(self).captured = {"url": url, "headers": headers}
        return _FakeHTTPResponse(
            '```json\n{"action": "done", "reasoning": "fenced", "summary": "ok"}\n```'
        )


def test_openai_compat_client_strips_code_fences(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr("httpx.Client", _FakeHTTPClient)
    client = OpenAICompatClient()
    req = ActionRequest(
        goal="g", step_index=0, history=[],
        observation={"url": "u", "title": "t", "a11y": "a", "dom_excerpt": "d"},
    )
    decision = client.decide(req)
    assert decision.action == "done"
    assert decision.summary == "ok"
    assert _FakeHTTPClient.captured["url"].endswith("/chat/completions")
    assert _FakeHTTPClient.captured["headers"]["Authorization"] == "Bearer test-key"


# ------------------------------------------------------------- surface.py ---


def test_build_strategy_chain_ordering():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            hints = {
                "text": "Log in",
                "css": "#login-btn",
                "xpath": "//button",
                "role": "button",
                "name": "Log in",
                "placeholder": "Search",
                "label": "Username",
            }
            chain = build_strategy_chain(hints, page)
            assert [s["type"] for s in chain] == ["role", "label", "placeholder", "css", "xpath", "text"]
            assert chain[0] == {"type": "role", "role": "button", "name": "Log in"}
            assert build_strategy_chain(None, page) == []
            assert build_strategy_chain({}, page) == []
            partial = build_strategy_chain({"text": "x", "css": "#y"}, page)
            assert [s["type"] for s in partial] == ["css", "text"]
        finally:
            browser.close()


def test_resolve_strategies_fallback(surface):
    strategies = [
        {"type": "css", "value": "#does-not-exist"},
        {"type": "text", "value": "Savings balance"},
    ]
    locator, index = resolve_strategies(surface.page, strategies)
    assert index == 1
    assert locator.is_visible()
    assert "Savings balance" in locator.inner_text()


def test_resolve_strategies_not_found(surface):
    with pytest.raises(TargetNotFoundError):
        resolve_strategies(surface.page, [{"type": "css", "value": "#nope"}])
    with pytest.raises(TargetNotFoundError):
        resolve_strategies(surface.page, [])


def test_surface_act_fill_click_read(surface):
    res = surface.act({"action": "fill", "target": {"text": "Username"}, "value": "operator"})
    assert res["ok"], res["detail"]
    assert surface.page.locator("#u").input_value() == "operator"

    res = surface.act({"action": "fill", "target": {"label": "Password"}, "value": "secret"})
    assert res["ok"], res["detail"]
    assert surface.page.locator("#p").input_value() == "secret"

    res = surface.act({"action": "click", "target": {"role": "button", "name": "Log in"}})
    assert res["ok"], res["detail"]
    assert res["strategy_index"] == 0

    res = surface.act({"action": "read", "target": {"text": "Savings balance"}})
    assert res["ok"] is True
    assert "$4,321.09" in res["text"]
    assert "Savings balance" in res["text"]


def test_surface_act_navigate_wait_press(surface):
    res = surface.act({"action": "wait", "ms": 50})
    assert res["ok"]

    res = surface.act({"action": "navigate", "value": data_url("<html><head><title>Two</title></head><body>hi</body></html>")})
    assert res["ok"]
    assert surface.page.title() == "Two"

    res = surface.act({"action": "press", "key": "Tab"})
    assert res["ok"]

    res = surface.act({"action": "bogus"})
    assert res["ok"] is False


def test_perceive_state(surface):
    state = surface.perceive(screenshot=False)
    assert isinstance(state, SurfaceState)
    assert state.title == "Mock Bank"
    assert state.url.startswith("data:text/html")
    assert "Savings balance" in state.dom_excerpt
    assert len(state.dom_excerpt) <= 4000
    assert state.screenshot_png is None
    logged = state.model_dump(mode="json")
    assert "screenshot_png" not in logged


# --------------------------------------------------------------- agent.py ---


def test_run_goal_completes_demo_script():
    url = data_url()
    surface = WebSurface()
    try:
        run = run_goal(
            goal="Look up member 12345 and read their current savings balance",
            entry_point=url,
            llm=ScriptedMockClient(default_demo_script(url)),
            surface=surface,
            timeout_s=120,
        )
    finally:
        surface.close()
    assert isinstance(run, AgentRun)
    assert run.status == "completed"
    assert run.ended_at is not None
    assert "$4,321.09" in (run.steps[-1].params.get("summary") or "")
    read_steps = [s for s in run.steps if s.action == "read"]
    assert len(read_steps) == 1
    assert "$4,321.09" in read_steps[0].params.get("read_text", "")
    assert all(s.target_rationale for s in run.steps if s.target_strategies)


def test_run_goal_stuck_on_repeated_clicks():
    url = data_url()
    script = [
        {"action": "click", "target": {"text": "Log in"}, "reasoning": "click again"},
    ] * 4 + [{"action": "done", "reasoning": "give up", "summary": "done"}]
    surface = WebSurface()
    try:
        run = run_goal(
            goal="click forever",
            entry_point=url,
            llm=ScriptedMockClient(script),
            surface=surface,
            timeout_s=120,
        )
    finally:
        surface.close()
    assert run.status == "stuck"
    assert len(run.steps) == 3


def test_run_goal_escalation_calls_manager():
    calls = []

    class FakeEscalation:
        def request_intervention(self, **kwargs):
            calls.append(kwargs)

    url = data_url()
    script = [{"action": "escalate", "reasoning": "blocked", "escalate_reason": "captcha"}]
    surface = WebSurface()
    try:
        run = run_goal(
            goal="do thing",
            entry_point=url,
            llm=ScriptedMockClient(script),
            surface=surface,
            escalation_mgr=FakeEscalation(),
            timeout_s=120,
        )
    finally:
        surface.close()
    assert run.status == "escalated"
    assert len(calls) == 1
    assert calls[0]["reason"] == "captcha"


class _BlockedPolicy:
    def check_action(self, action_dict, url):
        return {"verdict": "blocked"}


def test_run_goal_guardrail_blocked():
    url = data_url()
    surface = WebSurface()
    try:
        run = run_goal(
            goal="do thing",
            entry_point=url,
            llm=ScriptedMockClient(default_demo_script(url)),
            surface=surface,
            policy=_BlockedPolicy(),
            timeout_s=120,
        )
    finally:
        surface.close()
    assert run.status == "guardrail_blocked"


def test_run_goal_max_steps_is_stuck():
    url = data_url()
    script = [{"action": "wait", "ms": 1, "reasoning": "stall"}] * 10
    surface = WebSurface()
    try:
        run = run_goal(
            goal="stall out",
            entry_point=url,
            llm=ScriptedMockClient(script),
            surface=surface,
            max_steps=5,
            timeout_s=120,
        )
    finally:
        surface.close()
    assert run.status == "stuck"
