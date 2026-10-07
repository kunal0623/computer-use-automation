"""Tests for approval gating (stretch 1) and the capability catalog (stretch 2).

The approval gate in tools/replay.py runs before any browser work starts, so
the blocked-on-draft case is exercised at the CLI level without a browser.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bankgpt_cua.artifact import CapabilityArtifact
from bankgpt_cua.catalog import Catalog, CatalogError
from bankgpt_cua.registry import ApprovalError, Registry

BUILD_DIR = Path(__file__).resolve().parent.parent
VENV_PY = BUILD_DIR / ".venv" / "bin" / "python"


def _artifact(
    artifact_id="cap-test",
    version="1.0.0",
    name="Test Capability",
    approval_state="draft",
):
    return CapabilityArtifact(
        id=artifact_id,
        name=name,
        version=version,
        description="A test capability.",
        surface={"kind": "web", "entry_point": "https://bank.example/start", "notes": ""},
        inputs=[
            {
                "name": "member_id",
                "type": "string",
                "required": True,
                "pattern": r"\d{5}",
                "redact_in_logs": True,
            }
        ],
        outputs=[],
        steps=[],
        success_condition={
            "type": "text_present",
            "value": "done",
            "description": "test checkpoint",
        },
        error_policy=[],
        provenance={"run_id": "r1", "model": "test", "recorded_at": "t", "goal": "g"},
        approval_state=approval_state,
    )


def _write_artifact(tmp_path, artifact):
    path = tmp_path / "artifact.json"
    artifact.save(path)
    return path


# --- reliability scoring ---


def test_reliability_counts_business_outcomes_and_recovered_as_success(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    for cls in ["success", "success", "business_outcome", "recovered", "hard_failure"]:
        reg.record_replay("cap", "1.0.0", {"member_id": "12345"}, cls)
    record = reg.get("cap", "1.0.0")
    assert record is not None
    assert record.reliability() == pytest.approx(4 / 5)
    assert record.counts() == {
        "success": 2,
        "business_outcome": 1,
        "recovered": 1,
        "hard_failure": 1,
    }


def test_reliability_none_without_history(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register("cap", "1.0.0")
    assert reg.get("cap", "1.0.0").reliability() is None


def test_only_hashes_are_stored_never_raw_inputs(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.record_replay("cap", "1.0.0", {"member_id": "12345"}, "success")
    raw = (tmp_path / "registry.json").read_text(encoding="utf-8")
    assert "12345" not in raw
    record = reg.get("cap", "1.0.0")
    assert len(record.replays[0].inputs_hash) == 16


# --- approval lifecycle ---


def test_require_approved_refuses_draft_with_actionable_message(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register("cap", "1.0.0", artifact_path="/x/artifact.json")
    with pytest.raises(ApprovalError) as excinfo:
        reg.require_approved("cap", "1.0.0")
    message = str(excinfo.value)
    assert "not approved" in message
    assert "tools/approve.py approve" in message
    assert "/x/artifact.json" in message


def test_require_approved_refuses_unknown_artifact(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    with pytest.raises(ApprovalError):
        reg.require_approved("nope", "9.9.9")


def test_require_approved_passes_after_approve(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.approve("cap", "1.0.0", by="reviewer", notes="looks good")
    record = reg.require_approved("cap", "1.0.0")
    assert record.approval_state == "approved"
    assert record.approved_by == "reviewer"
    assert record.approved_at is not None


def test_reject_returns_to_draft_and_records_reason(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.approve("cap", "1.0.0", by="reviewer")
    record = reg.reject("cap", "1.0.0", by="reviewer", reason="flaky locator")
    assert record.approval_state == "draft"
    assert record.approved_by is None
    assert "flaky locator" in record.review_notes
    with pytest.raises(ApprovalError):
        reg.require_approved("cap", "1.0.0")


# --- tools/replay.py --unattended gate (CLI, no browser: gate is pre-surface) ---


def _run_cli(*argv, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [str(VENV_PY), *argv],
        cwd=BUILD_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_unattended_replay_blocked_on_draft(tmp_path):
    artifact_path = _write_artifact(tmp_path, _artifact(approval_state="draft"))
    registry_path = tmp_path / "registry.json"
    proc = _run_cli(
        "tools/replay.py",
        "--artifact", str(artifact_path),
        "--inputs", "member_id=12345",
        "--unattended",
        env_extra={"BANKGPT_REGISTRY": str(registry_path)},
    )
    assert proc.returncode == 3
    assert "not approved" in proc.stderr
    assert "tools/approve.py approve" in proc.stderr


def test_interactive_replay_warns_but_does_not_block_on_draft(tmp_path):
    # The gate must warn on stderr yet proceed; it will then fail fast at
    # browser startup (bogus entry point), so just assert the warning text
    # appears and the exit code is not the unattended-refusal code.
    artifact_path = _write_artifact(tmp_path, _artifact(approval_state="draft"))
    registry_path = tmp_path / "registry.json"
    proc = _run_cli(
        "tools/replay.py",
        "--artifact", str(artifact_path),
        "--inputs", "member_id=12345",
        "--out", str(tmp_path / "evidence"),
        env_extra={"BANKGPT_REGISTRY": str(registry_path)},
    )
    assert "not approved" in proc.stderr
    assert proc.returncode != 3  # 3 is reserved for the unattended refusal


# --- tools/approve.py ---


def test_approve_cli_mirrors_state_onto_artifact_and_registry(tmp_path):
    artifact_path = _write_artifact(tmp_path, _artifact(approval_state="draft"))
    registry_path = tmp_path / "registry.json"
    proc = _run_cli(
        "tools/approve.py", "approve",
        "--artifact", str(artifact_path),
        "--by", "reviewer",
        "--notes", "checked",
        env_extra={"BANKGPT_REGISTRY": str(registry_path)},
    )
    assert proc.returncode == 0, proc.stderr
    reloaded = CapabilityArtifact.load(artifact_path)
    assert reloaded.approval_state == "approved"
    assert reloaded.approved_by == "reviewer"
    assert reloaded.approved_at is not None
    reg = Registry(registry_path)
    assert reg.require_approved("cap-test", "1.0.0").approved_by == "reviewer"

    proc = _run_cli(
        "tools/approve.py", "status", "--artifact", str(artifact_path),
        env_extra={"BANKGPT_REGISTRY": str(registry_path)},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["approval_state"] == "approved"
    assert payload["reliability"] is None


# --- catalog ---


def _catalog(tmp_path, artifact):
    caps = tmp_path / "caps"
    caps.mkdir()
    artifact.save(caps / "cap.json")
    reg = Registry(tmp_path / "registry.json")
    return Catalog(capabilities_dir=caps, registry=reg), reg


def test_catalog_list_shows_approval_state_and_score(tmp_path):
    catalog, reg = _catalog(tmp_path, _artifact())
    listed = catalog.list_capabilities()
    assert len(listed) == 1
    entry = listed[0]
    assert entry["name"] == "Test Capability"
    assert entry["approval_state"] == "draft"
    assert entry["reliability"] is None
    assert entry["inputs"][0]["name"] == "member_id"

    reg.approve("cap-test", "1.0.0", by="reviewer")
    reg.record_replay("cap-test", "1.0.0", {"member_id": "1"}, "success")
    reg.record_replay("cap-test", "1.0.0", {"member_id": "2"}, "hard_failure")
    entry = catalog.list_capabilities()[0]
    assert entry["approval_state"] == "approved"
    assert entry["reliability"] == pytest.approx(0.5)
    assert entry["total_replays"] == 2


def test_catalog_invoke_refuses_unapproved(tmp_path):
    catalog, _ = _catalog(tmp_path, _artifact())
    with pytest.raises(CatalogError) as excinfo:
        catalog.invoke("Test Capability", {"member_id": "12345"})
    assert "not approved" in str(excinfo.value)


def test_catalog_invoke_validates_bad_inputs_before_browser(tmp_path):
    catalog, reg = _catalog(tmp_path, _artifact())
    reg.approve("cap-test", "1.0.0", by="reviewer")
    with pytest.raises(CatalogError) as excinfo:
        catalog.invoke("Test Capability", {})  # missing required member_id
    assert "member_id" in str(excinfo.value)
    with pytest.raises(CatalogError):
        catalog.invoke("Test Capability", {"member_id": "abc"})  # pattern


def test_catalog_get_unknown_and_ambiguous(tmp_path):
    catalog, _ = _catalog(tmp_path, _artifact())
    with pytest.raises(CatalogError) as excinfo:
        catalog.invoke("No Such Thing", {})
    assert "Unknown capability" in str(excinfo.value)


def test_catalog_list_ignores_unparseable_files(tmp_path):
    catalog, _ = _catalog(tmp_path, _artifact())
    (catalog.capabilities_dir / "broken.json").write_text("{not json",
                                                         encoding="utf-8")
    assert len(catalog.list_capabilities()) == 1
