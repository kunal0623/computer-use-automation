"""Tests for the capability artifact schema, error taxonomy, and replay engine."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from playwright.sync_api import sync_playwright
from pydantic import ValidationError

from bankgpt_cua import errors
from bankgpt_cua.artifact import (
    ArtifactBuilder,
    CapabilityArtifact,
    Checkpoint,
)
from bankgpt_cua.errors import OutcomeClass, classify
from bankgpt_cua.replay import ReplayResult, replay


# ---------------------------------------------------------------------------
# helpers


def _minimal_artifact_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "1.0",
        "id": "member-balance",
        "name": "Member savings balance lookup",
        "version": "1.0.0",
        "description": "Look up a member and read the savings balance.",
        "surface": {"kind": "web", "entry_point": "https://bank.example", "notes": ""},
        "inputs": [
            {
                "name": "member_id",
                "type": "string",
                "required": True,
                "pattern": r"\d{5}",
                "example": "12345",
            }
        ],
        "outputs": [],
        "steps": [
            {
                "index": 0,
                "action": "navigate",
                "target": None,
                "params": {"url": "${entry_point}/search"},
                "checkpoint": {
                    "type": "text_present",
                    "value": "Find member",
                    "description": "Search page loaded",
                },
                "notes": "",
            }
        ],
        "success_condition": {
            "type": "text_present",
            "value": "Savings balance",
            "description": "Balance visible at end",
        },
        "error_policy": [
            {
                "name": "member_not_found",
                "outcome_class": "business_outcome",
                "match": {"text_contains": ["No member found"]},
                "outcome": {
                    "code": "MEMBER_NOT_FOUND",
                    "message_template": "No member found for ID {member_id}",
                },
                "recovery": None,
            }
        ],
        "provenance": {
            "run_id": "run-1",
            "model": "test-model",
            "recorded_at": "2026-10-07",
            "goal": "read balance",
        },
        "review_notes": "",
    }
    base.update(overrides)
    return base


def _fake_run(**overrides: Any) -> SimpleNamespace:
    steps = [
        SimpleNamespace(
            index=0,
            action="navigate",
            target_strategies=[],
            target_rationale="",
            params={"url": "https://bank.example/start"},
            reasoning="Open the start page.",
            obs_before={"url": "", "title": ""},
            obs_after={"url": "https://bank.example/start", "title": "Start"},
            checkpoint=None,
        ),
        SimpleNamespace(
            index=1,
            action="click",
            target_strategies=[
                {"type": "role", "role": "button", "name": "Search member"},
                {"type": "text", "value": "Search member"},
            ],
            target_rationale="Role plus accessible name survives CSS churn.",
            params={},
            reasoning="Click the search button. " * 100,
            obs_before={"url": "https://bank.example/start", "title": "Start"},
            obs_after={
                "url": "https://bank.example/member",
                "title": "Member",
                "text": "Savings balance\n$1,234.56",
            },
            checkpoint={
                "type": "text_present",
                "value": "Savings balance",
                "description": "Balance shown",
            },
        ),
    ]
    run = SimpleNamespace(
        run_id="run-abc",
        goal="Read the savings balance for a member.",
        entry_point="https://bank.example/start",
        model_name="test-model",
        steps=steps,
        status="done",
        started_at="2026-10-07T10:00:00",
        ended_at="2026-10-07T10:01:00",
    )
    for k, v in overrides.items():
        setattr(run, k, v)
    return run


# ---------------------------------------------------------------------------
# 1. schema validation


def test_artifact_save_load_roundtrip(tmp_path):
    artifact = CapabilityArtifact.model_validate(_minimal_artifact_dict())
    path = tmp_path / "artifact.json"
    artifact.save(path)
    loaded = CapabilityArtifact.load(path)
    assert loaded == artifact
    assert loaded.schema_version == "1.0"
    assert loaded.version == "1.0.0"


def test_artifact_rejects_invalid_action():
    bad = _minimal_artifact_dict()
    bad["steps"][0]["action"] = "teleport"
    with pytest.raises(ValidationError):
        CapabilityArtifact.model_validate(bad)


def test_artifact_rejects_non_semver_version():
    bad = _minimal_artifact_dict(version="1.0")
    with pytest.raises(ValidationError):
        CapabilityArtifact.model_validate(bad)


def test_artifact_rejects_duplicate_step_indices():
    bad = _minimal_artifact_dict()
    bad["steps"].append(dict(bad["steps"][0]))
    with pytest.raises(ValidationError):
        CapabilityArtifact.model_validate(bad)


def test_artifact_json_schema():
    schema = CapabilityArtifact.json_schema()
    assert schema["title"] == "CapabilityArtifact"
    assert "ArtifactStep" in json.dumps(schema)


def test_to_summary_has_no_em_dashes():
    artifact = CapabilityArtifact.model_validate(_minimal_artifact_dict())
    summary = artifact.to_summary()
    assert "\u2014" not in summary
    assert "Member savings balance lookup" in summary
    assert "member_id" in summary
    assert "0. navigate" in summary
    assert "MEMBER_NOT_FOUND" in summary


# ---------------------------------------------------------------------------
# 2. builder from a recorded run


def test_builder_from_run():
    run = _fake_run()
    artifact = ArtifactBuilder.from_run(
        run,
        artifact_id="member-balance",
        name="Member savings balance lookup",
        inputs=[{"name": "member_id", "type": "string", "required": True}],
        outputs=[
            {
                "name": "balance",
                "type": "string",
                "extraction": {
                    "step_index": 1,
                    "strategy": {"type": "css", "value": "#balance"},
                    "postprocess": "strip_currency",
                },
            }
        ],
        error_policy=[
            {
                "name": "member_not_found",
                "outcome_class": "business_outcome",
                "match": {"text_contains": ["No member found"]},
                "outcome": {
                    "code": "MEMBER_NOT_FOUND",
                    "message_template": "No member found for ID {member_id}",
                },
                "recovery": None,
            }
        ],
        review_notes="Looks good.",
    )
    assert artifact.id == "member-balance"
    assert artifact.version == "1.0.0"
    assert artifact.schema_version == "2.0"
    assert len(artifact.steps) == 2
    assert [s.index for s in artifact.steps] == [0, 1]
    assert artifact.steps[0].target is None
    assert artifact.steps[0].action == "navigate"
    assert artifact.steps[1].target is not None
    assert len(artifact.steps[1].target.strategies) == 2
    assert (
        artifact.steps[1].target.rationale
        == "Role plus accessible name survives CSS churn."
    )
    assert artifact.steps[1].checkpoint is not None
    assert artifact.steps[1].checkpoint.type == "text_present"
    assert len(artifact.steps[1].notes) <= 500
    assert artifact.success_condition.type == "text_present"
    assert artifact.success_condition.value == "Savings balance"
    assert artifact.provenance["run_id"] == "run-abc"
    assert artifact.provenance["model"] == "test-model"
    assert artifact.surface["entry_point"] == "https://bank.example/start"
    assert artifact.review_notes == "Looks good."


def test_builder_raises_without_usable_final_state():
    run = _fake_run()
    for step in run.steps:
        step.obs_after = {"url": "", "title": ""}
    with pytest.raises(ValueError, match="no usable final state"):
        ArtifactBuilder.from_run(
            run,
            artifact_id="x",
            name="x",
            inputs=[],
            outputs=[],
            error_policy=[],
        )


# ---------------------------------------------------------------------------
# 3. error taxonomy


def test_classify_member_not_found():
    matches = classify("No member found for ID 99999", "https://bank.example/member")
    assert len(matches) == 1
    assert matches[0].name == "member_not_found"
    assert matches[0].outcome_class == OutcomeClass.BUSINESS_OUTCOME
    assert "No member found" in matches[0].evidence["text_excerpt"]


def test_classify_validation_error():
    matches = classify("Member ID must be 5 digits", "https://bank.example/search")
    assert [m.name for m in matches] == ["validation_error"]
    assert matches[0].outcome_class == OutcomeClass.BUSINESS_OUTCOME


def test_classify_session_expired_recoverable_with_recovery():
    matches = classify(
        "Session expired. Please log in again.", "https://bank.example/member"
    )
    assert len(matches) == 1
    assert matches[0].name == "session_expired"
    assert matches[0].outcome_class == OutcomeClass.RECOVERABLE
    assert matches[0].recovery is not None
    assert matches[0].recovery["action"] == "navigate"


def test_classify_permission_denied():
    matches = classify("Permission denied", "https://bank.example/admin")
    assert matches[0].name == "permission_denied"
    assert matches[0].outcome_class == OutcomeClass.BUSINESS_OUTCOME


def test_classify_app_error_hard_failure():
    matches = classify("Something went wrong", "https://bank.example/member")
    assert matches[0].name == "app_error"
    assert matches[0].outcome_class == OutcomeClass.HARD_FAILURE


def test_classify_is_case_insensitive_and_ordered():
    matches = classify("NO MEMBER FOUND for id 1", "https://bank.example/x")
    assert [m.name for m in matches] == ["member_not_found"]


def test_classify_no_match():
    assert classify("All good here", "https://bank.example/ok") == []


def test_classify_default_matchers_cover_mock_states():
    assert {m["name"] for m in errors.DEFAULT_MATCHERS} == {
        "member_not_found",
        "validation_error",
        "session_expired",
        "permission_denied",
        "app_error",
    }


# ---------------------------------------------------------------------------
# 4. replay against a local HTTP server


SEARCH_PAGE = """<html><head><title>Member search</title></head><body>
<h1>Find member</h1>
<a href="/account">View savings</a>
</body></html>"""

ACCOUNT_PAGE = """<html><head><title>Account</title></head><body>
<h1>Savings balance</h1>
<p id="balance">$1,234.56</p>
</body></html>"""

MISSING_PAGE = """<html><head><title>Missing</title></head><body>
<p>No member found for ID 99999</p>
</body></html>"""

EXPIRED_PAGE = """<html><head><title>Expired</title></head><body>
<p>Session expired. Please log in again.</p>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        pages = {
            "/search": SEARCH_PAGE,
            "/account": ACCOUNT_PAGE,
            "/missing": MISSING_PAGE,
            "/expired": EXPIRED_PAGE,
        }
        body = pages.get(self.path, "<html><body>not found</body></html>")
        code = 200 if self.path in pages else 404
        encoded = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def _pw_browser():
    """One shared Chromium for all replay tests.

    Playwright's sync API runs its event loop in the calling thread, so two
    sync_playwright() instances cannot be alive at once in one thread.
    """
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


class _TestSurface:
    """Minimal duck-typed surface for replay tests (real Chromium)."""

    def __init__(self, browser):
        self._browser = browser
        self._page = None

    def start(self, entry: str, headless: bool = True):
        self._page = self._browser.new_page()
        self._page.goto(entry)
        return self._page

    def close(self):
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None


class _FakeLogger:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.screenshots: list[tuple[str, bytes]] = []
        self.doms: list[tuple[str, str]] = []
        self.evidence_dir = "/tmp/evidence"

    def log(self, event: str, payload: dict):
        self.events.append((event, payload))

    def screenshot(self, name: str, png_bytes: bytes):
        self.screenshots.append((name, png_bytes))

    def dom(self, name: str, html: str):
        self.doms.append((name, html))


class _FakeEscalation:
    def __init__(self):
        self.calls: list[tuple] = []

    def request_intervention(
        self, run_id, goal, step_index, reason, state_summary, screenshot
    ):
        self.calls.append(
            (run_id, goal, step_index, reason, state_summary, screenshot)
        )


def _balance_artifact(entry_point: str) -> CapabilityArtifact:
    return CapabilityArtifact.model_validate(
        _minimal_artifact_dict(
            surface={"kind": "web", "entry_point": entry_point, "notes": ""},
            inputs=[],
            outputs=[
                {
                    "name": "balance",
                    "type": "string",
                    "description": "Savings balance without currency formatting.",
                    "extraction": {
                        "step_index": 2,
                        "strategy": {"type": "css", "value": "#balance"},
                        "postprocess": "strip_currency",
                    },
                }
            ],
            steps=[
                {
                    "index": 0,
                    "action": "navigate",
                    "target": None,
                    "params": {"url": "${entry_point}/search"},
                    "checkpoint": {
                        "type": "text_present",
                        "value": "Find member",
                        "description": "Search page loaded",
                    },
                    "notes": "",
                },
                {
                    "index": 1,
                    "action": "click",
                    "target": {
                        "strategies": [
                            {
                                "type": "role",
                                "role": "link",
                                "name": "View savings",
                            },
                            {"type": "text", "value": "View savings"},
                        ],
                        "rationale": "Role plus accessible name survives CSS churn.",
                    },
                    "params": {},
                    "checkpoint": {
                        "type": "text_present",
                        "value": "Savings balance",
                        "description": "Account page loaded",
                    },
                    "notes": "",
                },
                {
                    "index": 2,
                    "action": "read",
                    "target": {
                        "strategies": [{"type": "css", "value": "#balance"}],
                        "rationale": "Stable id on the balance element.",
                    },
                    "params": {"name": "balance_raw", "postprocess": "strip_currency"},
                    "checkpoint": None,
                    "notes": "",
                },
            ],
        )
    )


def test_replay_success_and_deterministic(server, _pw_browser):
    artifact = _balance_artifact(server)
    logger = _FakeLogger()
    first = replay(artifact, {}, _TestSurface(_pw_browser), logger=logger)
    second = replay(artifact, {}, _TestSurface(_pw_browser), logger=_FakeLogger())
    assert isinstance(first, ReplayResult)
    assert first.status == "success"
    assert first.outputs["balance"] == "1234.56"
    assert first.outputs["balance_raw"] == "1234.56"
    assert first.steps_completed == 3
    assert first.recoveries_applied == []
    assert first.failure is None
    # determinism: same artifact and inputs give the same run id and outputs
    assert second.run_id == first.run_id
    assert second.outputs == first.outputs
    assert second.status == first.status
    # winning strategy index was logged
    wins = [
        p["winning_strategy"]
        for e, p in logger.events
        if e == "strategy_resolved"
    ]
    assert wins == [0, 0]


def test_replay_invalid_input_returns_business_outcome_without_browser():
    artifact = CapabilityArtifact.model_validate(_minimal_artifact_dict())

    class _ExplodingSurface:
        def start(self, *a, **k):
            raise AssertionError("browser must not start on invalid input")

    result = replay(artifact, {}, _ExplodingSurface())
    assert result.status == "business_outcome"
    assert result.business_outcome["code"] == "INVALID_INPUT"
    assert "member_id" in result.business_outcome["message"]
    assert result.steps_completed == 0

    bad_pattern = replay(artifact, {"member_id": "12"}, _ExplodingSurface())
    assert bad_pattern.business_outcome["code"] == "INVALID_INPUT"

    ok_coerced = replay(
        CapabilityArtifact.model_validate(
            _minimal_artifact_dict(
                inputs=[{"name": "n", "type": "integer", "required": True}],
                steps=[],
                success_condition={
                    "type": "text_present",
                    "value": "x",
                    "description": "x",
                },
            )
        ),
        {"n": "42"},
        _ExplodingSurface(),
    )
    # integer coercion passes validation; failure is at surface start, not input
    assert ok_coerced.failure["code"] == "SURFACE_START_FAILED"


def test_replay_business_outcome_renders_template(server, _pw_browser):
    artifact = CapabilityArtifact.model_validate(
        _minimal_artifact_dict(
            surface={"kind": "web", "entry_point": server, "notes": ""},
            inputs=[
                {"name": "member_id", "type": "string", "required": True},
            ],
            steps=[
                {
                    "index": 0,
                    "action": "navigate",
                    "target": None,
                    "params": {"url": "${entry_point}/missing"},
                    "checkpoint": None,
                    "notes": "",
                }
            ],
            success_condition={
                "type": "text_present",
                "value": "Savings balance",
                "description": "unused",
            },
        )
    )
    surface = _TestSurface(_pw_browser)
    result = replay(artifact, {"member_id": "99999"}, surface)
    assert result.status == "business_outcome"
    assert result.business_outcome["code"] == "MEMBER_NOT_FOUND"
    assert result.business_outcome["message"] == "No member found for ID 99999"


def test_replay_checkpoint_miss_escalates_and_captures_evidence(server, _pw_browser):
    artifact = CapabilityArtifact.model_validate(
        _minimal_artifact_dict(
            surface={"kind": "web", "entry_point": server, "notes": ""},
            inputs=[],
            steps=[
                {
                    "index": 0,
                    "action": "navigate",
                    "target": None,
                    "params": {"url": "${entry_point}/search"},
                    "checkpoint": {
                        "type": "text_present",
                        "value": "text that never appears",
                        "description": "impossible checkpoint",
                    },
                    "notes": "",
                }
            ],
        )
    )
    surface = _TestSurface(_pw_browser)
    logger = _FakeLogger()
    escalation = _FakeEscalation()
    result = replay(artifact, {}, surface, logger=logger, escalation_mgr=escalation)
    assert result.status == "hard_failure"
    assert result.failure["code"] == "CHECKPOINT_MISS"
    assert result.failure["step_index"] == 0
    assert "text that never appears" in result.failure["expected"]
    assert logger.screenshots, "expected a failure screenshot"
    assert logger.doms, "expected failure DOM capture"
    assert len(escalation.calls) == 1
    call = escalation.calls[0]
    assert call[0] == result.run_id
    assert call[2] == 0
    assert isinstance(call[5], (bytes, type(None)))


def test_replay_guardrail_denial(server, _pw_browser):
    artifact = _balance_artifact(server)
    policy = SimpleNamespace(
        check_action=lambda action, url: {
            "allowed": False,
            "requires_human": True,
            "reason": "test denial",
        }
    )
    escalation = _FakeEscalation()
    surface = _TestSurface(_pw_browser)
    result = replay(artifact, {}, surface, policy=policy, escalation_mgr=escalation)
    assert result.status == "hard_failure"
    assert result.failure["code"] == "GUARDRAIL_DENIED"
    assert "test denial" in result.failure["observed"]
    assert len(escalation.calls) == 1


def test_replay_recoverable_session_expired_applies_recovery(server, _pw_browser):
    artifact = CapabilityArtifact.model_validate(
        _minimal_artifact_dict(
            surface={"kind": "web", "entry_point": server, "notes": ""},
            inputs=[],
            steps=[
                {
                    "index": 0,
                    "action": "navigate",
                    "target": None,
                    "params": {"url": "${entry_point}/expired"},
                    "checkpoint": None,
                    "notes": "",
                }
            ],
            success_condition={
                "type": "text_present",
                "value": "Find member",
                "description": "Back on the search page after re-login",
            },
            error_policy=[
                {
                    "name": "session_expired",
                    "outcome_class": "recoverable",
                    "match": {"text_contains": ["Session expired"]},
                    "outcome": {
                        "code": "SESSION_EXPIRED",
                        "message_template": "Session expired; reloaded entry point.",
                    },
                    "recovery": {
                        "action": "navigate",
                        "params": {"url": "${entry_point}/search"},
                    },
                }
            ],
        )
    )
    logger = _FakeLogger()
    result = replay(artifact, {}, _TestSurface(_pw_browser), logger=logger)
    assert result.status == "recovered"
    assert result.recoveries_applied == ["session_expired"]
    assert result.steps_completed == 1
    assert any(e == "recovery_applied" for e, _ in logger.events)
