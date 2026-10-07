"""Tests for guardrails, escalation/handoff, and evidence logging."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from bankgpt_cua.guardrails import (
    Policy,
    check_action,
    redact,
    redact_dict,
)
from bankgpt_cua.escalation import (
    ControlState,
    EscalationManager,
    collect_human_actions,
    create_operator_app,
    inject_recorder,
    set_control,
)
from bankgpt_cua.evidence import RunLogger, new_run_dir

BUILD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


# ---------------------------------------------------------------- guardrails


def test_navigate_inside_allowlist_is_allowed():
    policy = Policy()
    action = {"action": "navigate", "value": "http://127.0.0.1:8765/login"}
    result = check_action(action, "http://127.0.0.1:8765/", policy)
    assert result["allowed"] is True
    assert result["requires_human"] is False


def test_navigate_outside_allowlist_is_blocked():
    policy = Policy()
    action = {"action": "navigate", "value": "https://evil.example.com/phish"}
    result = check_action(action, "http://127.0.0.1:8765/", policy)
    assert result["allowed"] is False
    assert "allowlist" in result["reason"]


def test_navigate_blocked_url_pattern():
    policy = Policy(blocked_url_patterns=[r"logout"])
    action = {"action": "navigate", "value": "http://127.0.0.1:8765/logout"}
    result = check_action(action, "http://127.0.0.1:8765/", policy)
    assert result["allowed"] is False
    assert "blocked" in result["reason"]


def test_disallowed_action_type_is_blocked():
    policy = Policy()
    action = {"action": "download", "value": "http://127.0.0.1:8765/x"}
    result = check_action(action, "http://127.0.0.1:8765/", policy)
    assert result["allowed"] is False
    assert "download" in result["reason"]


def test_risky_click_blocked_by_default():
    policy = Policy()  # risky_mode="block"
    action = {
        "action": "click",
        "target": {"text": "Close savings account", "role": "button"},
    }
    result = check_action(action, "http://127.0.0.1:8765/account", policy)
    assert result["allowed"] is False
    assert result["requires_human"] is False
    assert "Close savings account" in result["reason"]


def test_risky_click_confirm_mode_requires_human():
    policy = Policy(risky_mode="confirm")
    action = {
        "action": "click",
        "target": {"text": "Close savings account", "role": "button"},
    }
    result = check_action(action, "http://127.0.0.1:8765/account", policy)
    assert result["allowed"] is True
    assert result["requires_human"] is True
    assert "confirmation" in result["reason"]


def test_irreversible_param_is_risky():
    policy = Policy(risky_mode="confirm")
    action = {
        "action": "click",
        "target": {"text": "Submit transfer"},
        "params": {"irreversible": True},
    }
    result = check_action(action, "http://127.0.0.1:8765/transfer", policy)
    assert result["requires_human"] is True


def test_benign_click_allowed():
    policy = Policy()
    action = {"action": "click", "target": {"text": "Search member"}}
    result = check_action(action, "http://127.0.0.1:8765/members", policy)
    assert result == {"allowed": True, "requires_human": False, "reason": "ok"}


# ---------------------------------------------------------------- redaction


def test_redact_ssn():
    assert redact("ssn 123-45-6789 here") == "ssn [REDACTED:ssn] here"


def test_redact_credit_card():
    assert redact("card 4111 1111 1111 1111 ok") == "card [REDACTED:card] ok"
    assert redact("card 4111-1111-1111-1111 ok") == "card [REDACTED:card] ok"


def test_redact_bearer_token():
    out = redact("Authorization: Bearer abcDEF123-._~+/xyz")
    assert "Bearer abcDEF" not in out
    assert "[REDACTED:bearer]" in out


def test_redact_password_assignment():
    out = redact("login with password: hunter2 now")
    assert "hunter2" not in out
    assert "password: [REDACTED:password]" in out


def test_redact_extra_values():
    out = redact("member id is M-99881", extra_values=["M-99881"])
    assert out == "member id is [REDACTED:param]"


def test_redact_dict_recursive():
    payload = {
        "user": "kunal",
        "ssn": "123-45-6789",
        "nested": [{"token": "Bearer abc123"}],
        "count": 3,
    }
    out = redact_dict(payload)
    assert out["user"] == "kunal"
    assert out["ssn"] == "[REDACTED:ssn]"
    assert out["nested"][0]["token"] == "[REDACTED:bearer]"
    assert out["count"] == 3


# ---------------------------------------------------------------- escalation


def test_initial_state_is_auto():
    mgr = EscalationManager()
    assert mgr.state is ControlState.AUTO


def test_request_intervention_pauses():
    mgr = EscalationManager()
    req = mgr.request_intervention(
        run_id="r1", goal="check balance", step_index=3,
        reason="risky click", state_summary="on account page", screenshot=None,
    )
    assert mgr.state is ControlState.PAUSED
    assert req.status == "open"
    assert req.run_id == "r1"


def test_takeover_and_handback_cycle():
    mgr = EscalationManager()
    mgr.request_intervention("r1", "g", 1, "why", "summary", None)
    mgr.takeover()
    assert mgr.state is ControlState.HUMAN
    mgr.handback()
    assert mgr.state is ControlState.AUTO


def test_illegal_transitions_raise():
    mgr = EscalationManager()
    with pytest.raises(RuntimeError):
        mgr.takeover()  # AUTO -> HUMAN is illegal
    with pytest.raises(RuntimeError):
        mgr.handback()  # AUTO -> AUTO is illegal
    mgr.request_intervention("r1", "g", 1, "why", "summary", None)
    with pytest.raises(RuntimeError):
        mgr.handback()  # PAUSED -> AUTO is illegal


def test_intervention_saves_screenshot(tmp_path):
    mgr = EscalationManager(evidence_dir=str(tmp_path))
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    req = mgr.request_intervention("r1", "g", 1, "why", "summary", png)
    assert req.screenshot_path is not None
    assert os.path.isfile(req.screenshot_path)
    with open(req.screenshot_path, "rb") as handle:
        assert handle.read() == png


def test_record_human_action():
    mgr = EscalationManager()
    mgr.record_human_action("clicked submit", {"target": "button#submit"})
    actions = mgr.human_actions
    assert len(actions) == 1
    assert actions[0]["description"] == "clicked submit"
    assert actions[0]["detail"] == {"target": "button#submit"}


class _FakePage:
    """Minimal Playwright page double for recorder tests."""

    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.evaluated: list[str] = []
        self._buffer: list[dict] = [{"kind": "click", "target": "button#x"}]

    def add_init_script(self, js: str) -> None:
        self.scripts.append(js)

    def evaluate(self, js: str):
        self.evaluated.append(js)
        if "__bankgpt_human_actions = []" in js:
            buffered, self._buffer = self._buffer, []
            return buffered
        return None


def test_inject_recorder_and_control_helpers():
    page = _FakePage()
    inject_recorder(page)
    assert len(page.scripts) == 1
    assert "__bankgpt_human_actions" in page.scripts[0]
    assert '__bankgpt_control !== "HUMAN"' in page.scripts[0]
    set_control(page, ControlState.HUMAN)
    assert "HUMAN" in page.evaluated[-1]
    actions = collect_human_actions(page)
    assert actions == [{"kind": "click", "target": "button#x"}]
    assert collect_human_actions(page) == []  # buffer was cleared


def test_operator_console_api():
    mgr = EscalationManager()
    mgr.request_intervention("r1", "check balance", 2, "risky click",
                             "on the account page", None)
    app = create_operator_app(mgr, {"goal": "check balance", "run_dir": "/tmp/x"})
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "Control state" in page.text
    assert "check balance" in page.text

    state = client.get("/api/state").json()
    assert state["state"] == "PAUSED"

    # Illegal takeover path returns 409 from a fresh AUTO manager.
    fresh = TestClient(create_operator_app(EscalationManager(), {}))
    bad = fresh.post("/api/takeover")
    assert bad.status_code == 409

    ok = client.post("/api/takeover")
    assert ok.status_code == 200
    assert ok.json()["state"] == "HUMAN"

    mgr.record_human_action("typed member id", {})
    actions = client.get("/api/human_actions").json()
    assert len(actions["human_actions"]) == 1

    back = client.post("/api/handback")
    assert back.status_code == 200
    assert back.json()["state"] == "AUTO"


# ---------------------------------------------------------------- evidence


def test_new_run_dir_creates_unique_dirs(tmp_path):
    first = new_run_dir(str(tmp_path), "discover")
    second = new_run_dir(str(tmp_path), "discover")
    assert os.path.isdir(first)
    assert os.path.isdir(second)
    assert first != second


def test_logger_redacts_secrets_in_jsonl(tmp_path):
    run_dir = new_run_dir(str(tmp_path), "run")
    logger = RunLogger(run_dir)
    logger.log("step", {"action": "fill password: hunter2", "ssn": "123-45-6789"})
    logger.finalize("run-1")

    with open(os.path.join(run_dir, "run.jsonl"), encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "step"
    assert "hunter2" not in record["payload"]["action"]
    assert "[REDACTED:password]" in record["payload"]["action"]
    assert record["payload"]["ssn"] == "[REDACTED:ssn]"


def test_logger_screenshots_dom_and_manifest(tmp_path):
    run_dir = new_run_dir(str(tmp_path), "run")
    logger = RunLogger(run_dir)
    png_path = logger.screenshot("step1", b"\x89PNG" + b"\x00" * 8)
    dom_path = logger.dom("step1", "<html></html>")
    assert os.path.isfile(png_path)
    assert os.path.isfile(dom_path)
    logger.log("done", {})
    manifest = logger.finalize("run-9")
    assert manifest["run_id"] == "run-9"
    assert manifest["events"] == 1
    assert len(manifest["files"]) == 2
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as handle:
        assert json.load(handle) == manifest


# ---------------------------------------------------------------- CLI wiring


@pytest.mark.parametrize("tool", ["serve_mock.py", "discover.py", "replay.py"])
def test_cli_help_exits_zero(tool):
    proc = subprocess.run(
        [PYTHON, os.path.join(BUILD_DIR, "tools", tool), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
