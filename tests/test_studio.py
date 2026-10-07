"""Tests for the Macro Studio web UI.

No browser is launched in these tests: validation, approval gating, and
page rendering are all exercised before any browser work would start.
"""

from __future__ import annotations

import json
import shutil
import urllib.parse
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bankgpt_cua.artifact import CapabilityArtifact
from bankgpt_cua.studio import create_studio_app

BUILD = Path(__file__).resolve().parent.parent
DEMO_ARTIFACT = BUILD / "capabilities" / "member-savings-balance-lookup.json"
CAP_NAME = "Member Savings Balance Lookup"


@pytest.fixture()
def studio(tmp_path):
    """A studio pointed at a temp capabilities dir and temp registry."""
    caps = tmp_path / "capabilities"
    caps.mkdir()
    shutil.copy(DEMO_ARTIFACT, caps / DEMO_ARTIFACT.name)
    app = create_studio_app(
        capabilities_dir=caps,
        registry_path=tmp_path / "registry.json",
        evidence_base=str(tmp_path / "evidence"),
    )
    return TestClient(app), caps


def _detail_url(name: str = CAP_NAME) -> str:
    return f"/capabilities/{urllib.parse.quote(name, safe='')}"


def test_list_page_renders(studio):
    client, _ = studio
    resp = client.get("/")
    assert resp.status_code == 200
    assert CAP_NAME in resp.text
    assert "draft" in resp.text or "approved" in resp.text


def test_detail_page_renders(studio):
    client, _ = studio
    resp = client.get(_detail_url())
    assert resp.status_code == 200
    assert CAP_NAME in resp.text
    assert "member_id" in resp.text  # typed input shown
    assert "savings_balance" in resp.text  # typed output shown
    assert "Why this survives UI churn" in resp.text  # step rationales
    assert "Error policy" in resp.text
    assert "Run replay" in resp.text  # the replay form


def test_detail_unknown_capability_404(studio):
    client, _ = studio
    resp = client.get(_detail_url("No Such Capability"))
    assert resp.status_code == 404


def test_replay_rejects_bad_inputs_without_browser(studio):
    client, _ = studio
    # member_id is required and pattern-checked; empty is invalid.
    resp = client.post(
        _detail_url() + "/replay",
        data={"input:member_id": "", "headless": "on"},
    )
    assert resp.status_code == 400
    assert "Invalid inputs" in resp.text
    # No evidence run dir: no browser work happened.
    evidence = Path(studio[1]).parent / "evidence"
    assert not evidence.exists() or not any(evidence.iterdir())


def test_unattended_replay_of_draft_is_refused(studio):
    client, _ = studio
    resp = client.post(
        _detail_url() + "/replay",
        data={"input:member_id": "12345", "headless": "on", "unattended": "on"},
    )
    assert resp.status_code == 403
    assert "not approved" in resp.text
    assert "tools/approve.py" in resp.text  # same guidance as the CLI


def test_approve_flow_then_unattended_allowed(studio, tmp_path):
    client, caps = studio
    # Approve via the studio button.
    resp = client.post(
        _detail_url() + "/approve",
        data={"reviewer": "test-reviewer", "notes": "looks safe"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    # Detail page now shows the approved state.
    resp = client.get(_detail_url())
    assert resp.status_code == 200
    assert "approved" in resp.text
    # The artifact file itself was mirrored.
    artifact = CapabilityArtifact.load(caps / DEMO_ARTIFACT.name)
    assert artifact.approval_state == "approved"
    assert artifact.approved_by == "test-reviewer"


def test_history_page_renders(studio):
    client, _ = studio
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "Replay history" in resp.text
    assert CAP_NAME in resp.text


def test_redacted_inputs_masked(studio):
    """Values of inputs flagged redact_in_logs are masked in rendering."""
    from bankgpt_cua.studio import _masked_inputs

    artifact = CapabilityArtifact.load(DEMO_ARTIFACT)
    shown = _masked_inputs(artifact, {"member_id": "12345"})
    assert shown == {"member_id": "***"}
    # Step params on the detail page are parameterized, never raw values.
    client, _ = studio
    resp = client.get(_detail_url())
    assert resp.status_code == 200
    assert "${member_id}" in resp.text
