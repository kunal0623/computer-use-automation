"""Tests for the Studio goal-chat page.

No browser is launched and no LLM is called: the agent-loop entry point
and the surface factory are monkeypatched with fakes, so jobs run
deterministically in-process. What is exercised: request validation, the
concurrency cap, the SSE stream contract, screenshot routing, and the
save-as-capability flow.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bankgpt_cua import goals
from bankgpt_cua.agent import AgentRun, RecordedStep
from bankgpt_cua.studio import create_studio_app


def _fake_steps():
    steps = []
    for i, action in enumerate(("navigate", "click", "done")):
        steps.append(
            RecordedStep(
                index=i,
                action=action,
                target_strategies=[],
                target_rationale="",
                params={},
                reasoning=f"fake reasoning for step {i}",
                obs_before={"url": "http://127.0.0.1:8765/login", "title": "t"},
                obs_after={
                    "url": "http://127.0.0.1:8765/member/12345",
                    "title": "MockBank Member 12345",
                },
            )
        )
    return steps


def _fake_run_goal(
    goal,
    entry_point,
    llm,
    surface,
    *,
    policy=None,
    escalation_mgr=None,
    logger=None,
    max_steps=25,
    timeout_s=600,
):
    steps = _fake_steps()
    for rec in steps:
        if logger is not None:
            logger.log("step", {"run_id": "fakerun", "step": rec.model_dump()})
    return AgentRun(
        run_id="fakerun",
        goal=goal,
        entry_point=entry_point,
        model_name="fake",
        steps=steps,
        status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
    )


class _DummySurface:
    page = None

    def start(self, entry_url, headless=True):
        pass

    def close(self):
        pass


@pytest.fixture()
def chat_client(tmp_path, monkeypatch):
    """A studio app with fakes: no browser, no LLM, mock bank 'reachable'."""
    monkeypatch.setattr(goals, "RUN_GOAL_FN", _fake_run_goal)
    monkeypatch.setattr(goals, "SURFACE_FACTORY", lambda headless: _DummySurface())
    monkeypatch.setattr(
        "bankgpt_cua.studio._check_entry_point", lambda url, timeout=3.0: True
    )
    caps = tmp_path / "capabilities"
    caps.mkdir()
    app = create_studio_app(
        capabilities_dir=caps,
        registry_path=tmp_path / "registry.json",
        evidence_base=str(tmp_path / "evidence"),
    )
    return TestClient(app), caps


def _start(http, **overrides):
    payload = {"goal": "look up member 12345", "client": "mock", "headless": True}
    payload.update(overrides)
    return http.post("/api/goals", json=payload)


def _wait_terminal(client, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/goals/{job_id}")
        assert resp.status_code == 200
        if resp.json()["status"] != "running":
            return resp.json()
        time.sleep(0.05)
    raise AssertionError("job did not reach a terminal status in time")


def test_chat_page_renders(chat_client):
    client, _ = chat_client
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert "Goal chat" in resp.text
    assert 'id="goal"' in resp.text
    assert "/api/goals" in resp.text


def test_start_job_with_mock_client(chat_client):
    client, _ = chat_client
    resp = _start(client)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id
    final = _wait_terminal(client, job_id)
    assert final["status"] == "completed"
    assert final["artifact_id"]


def test_empty_goal_rejected(chat_client):
    client, _ = chat_client
    resp = _start(client, goal="   ")
    assert resp.status_code == 400
    assert "empty" in resp.json()["error"].lower()


def test_unknown_client_rejected(chat_client):
    client, _ = chat_client
    resp = _start(client, client="nope")
    assert resp.status_code == 400


def test_openai_without_key_rejected(chat_client, monkeypatch):
    client, _ = chat_client
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = _start(client, client="openai")
    assert resp.status_code == 400
    assert "API key" in resp.json()["error"]


def test_second_job_rejected_while_running(chat_client, monkeypatch):
    client, _ = chat_client
    gate = threading.Event()

    def blocking_fake(*args, **kwargs):
        assert gate.wait(timeout=15)
        return _fake_run_goal(*args, **kwargs)

    monkeypatch.setattr(goals, "RUN_GOAL_FN", blocking_fake)
    first = _start(client)
    assert first.status_code == 200
    second = _start(client)
    assert second.status_code == 409
    assert "already in progress" in second.json()["error"]
    gate.set()
    _wait_terminal(client, first.json()["job_id"])


def test_unknown_job_404(chat_client):
    client, _ = chat_client
    assert client.get("/api/goals/nope").status_code == 404
    assert client.get("/api/goals/nope/events").status_code == 404
    assert client.post("/api/goals/nope/save").status_code == 404


def _drain_sse(client, job_id):
    """Read the SSE stream to the final event; return (steps, final)."""
    steps, final = [], None
    with client.stream("GET", f"/api/goals/{job_id}/events") as resp:
        assert resp.status_code == 200
        buf = ""
        for chunk in resp.iter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                if raw.startswith(":") or not raw.strip():
                    continue
                event, data = None, None
                for line in raw.splitlines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                if event == "step":
                    steps.append(json.loads(data))
                elif event == "final":
                    final = json.loads(data)
                    break
            if final is not None:
                break
    return steps, final


def test_sse_stream_terminates_with_final(chat_client):
    client, _ = chat_client
    job_id = _start(client).json()["job_id"]
    steps, final = _drain_sse(client, job_id)
    assert final is not None
    assert final["status"] == "completed"
    assert final["artifact_id"]
    assert len(steps) == 3
    first = steps[0]
    assert first["index"] == 0
    assert first["action"] == "navigate"
    assert "fake reasoning" in first["reasoning"]
    # Dummy surface has no page, so no screenshots in this test.
    assert first["shot"] is None


def test_shot_missing_is_404(chat_client):
    client, _ = chat_client
    job_id = _start(client).json()["job_id"]
    _wait_terminal(client, job_id)
    resp = client.get(f"/api/goals/{job_id}/shots/0")
    assert resp.status_code == 404


def test_save_as_capability_copies_artifact(chat_client):
    client, caps = chat_client
    job_id = _start(client).json()["job_id"]
    _wait_terminal(client, job_id)
    resp = client.post(f"/api/goals/{job_id}/save")
    assert resp.status_code == 200
    body = resp.json()
    assert body["url"].startswith("/capabilities/")
    saved = Path(body["path"])
    assert saved.is_file()
    assert saved.parent == caps
    # The catalog picks it up on the list page.
    listing = client.get("/")
    assert listing.status_code == 200
    assert "look up member 12345" in listing.text


def test_save_collision_suffixes(chat_client):
    client, caps = chat_client
    job_id = _start(client).json()["job_id"]
    _wait_terminal(client, job_id)
    first = client.post(f"/api/goals/{job_id}/save").json()["path"]
    second = client.post(f"/api/goals/{job_id}/save").json()["path"]
    assert first != second
    assert second.endswith("-2.json")
    assert Path(first).is_file() and Path(second).is_file()


def test_save_before_completion_rejected(chat_client, monkeypatch):
    client, _ = chat_client
    gate = threading.Event()

    def blocking_fake(*args, **kwargs):
        assert gate.wait(timeout=15)
        return _fake_run_goal(*args, **kwargs)

    monkeypatch.setattr(goals, "RUN_GOAL_FN", blocking_fake)
    job_id = _start(client).json()["job_id"]
    resp = client.post(f"/api/goals/{job_id}/save")
    assert resp.status_code == 400
    gate.set()
    _wait_terminal(client, job_id)
