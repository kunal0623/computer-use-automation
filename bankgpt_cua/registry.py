"""Local operational registry for capability artifacts.

The registry is the operational record of a capability's lifecycle: its
approval state and its replay history. It is local machine state (gitignored),
separate from the artifact file itself. tools/approve.py mirrors approval
fields onto the artifact file so the artifact stays self-describing, but the
registry is the source of truth for unattended execution.

Reliability scoring: reliability = successful replays / total replays, where
"successful" means the capability executed correctly. That includes
business outcomes (e.g. MEMBER_NOT_FOUND is a correct answer to the caller's
question, not a failure) and recovered runs (the goal was reached after a
bounded recovery). Only hard failures count against reliability, because only
they mean the capability did not do its job.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ResultClass = Literal["success", "business_outcome", "recovered", "hard_failure"]

# Result classes that count as successful executions for reliability scoring.
# See the module docstring for the reasoning.
SUCCESSFUL_CLASSES = frozenset({"success", "business_outcome", "recovered"})


class ReplayRecord(BaseModel):
    """One recorded replay of a capability."""

    timestamp: str = Field(description="ISO UTC timestamp of the replay.")
    inputs_hash: str = Field(
        description="sha256 (truncated) of the canonicalized inputs. "
        "The hash is stored so raw input values, which may be sensitive, "
        "never land in the registry."
    )
    result_class: ResultClass


class ArtifactRecord(BaseModel):
    """Lifecycle and history record for one artifact id + version."""

    artifact_id: str
    version: str
    artifact_path: str = ""
    approval_state: Literal["draft", "approved", "deprecated"] = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    review_notes: str = ""
    first_seen: str = ""
    updated_at: str = ""
    replays: list[ReplayRecord] = Field(default_factory=list)

    def reliability(self) -> float | None:
        """Fraction of replays that executed successfully.

        Returns None when there is no replay history yet. Business outcomes
        and recovered runs count as successful; hard failures do not.
        """
        if not self.replays:
            return None
        ok = sum(1 for r in self.replays if r.result_class in SUCCESSFUL_CLASSES)
        return ok / len(self.replays)

    def counts(self) -> dict[str, int]:
        """Replay counts keyed by result class."""
        out: dict[str, int] = {
            "success": 0,
            "business_outcome": 0,
            "recovered": 0,
            "hard_failure": 0,
        }
        for r in self.replays:
            out[r.result_class] = out.get(r.result_class, 0) + 1
        return out


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_registry_path() -> Path:
    """Registry location; override with BANKGPT_REGISTRY."""
    return Path(os.environ.get("BANKGPT_REGISTRY", "./registry.json"))


def hash_inputs(inputs: dict[str, Any]) -> str:
    """Stable truncated hash of inputs. Values are never persisted."""
    canonical = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ApprovalError(RuntimeError):
    """Raised when an unattended run needs an approval it does not have."""


class Registry:
    """JSON-backed registry of artifact lifecycle and replay history."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_registry_path()
        self._records: dict[str, ArtifactRecord] = {}
        self._load()

    @staticmethod
    def _key(artifact_id: str, version: str) -> str:
        return f"{artifact_id}@{version}"

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raw = {}
        records = raw.get("records", {}) if isinstance(raw, dict) else {}
        for key, rec in records.items():
            try:
                self._records[key] = ArtifactRecord.model_validate(rec)
            except Exception:
                continue  # skip corrupt entries; the registry must never crash a run

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": {k: v.model_dump() for k, v in self._records.items()}
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, artifact_id: str, version: str) -> ArtifactRecord | None:
        """Return the record for an artifact, or None if never registered."""
        return self._records.get(self._key(artifact_id, version))

    def register(
        self,
        artifact_id: str,
        version: str,
        artifact_path: str = "",
        approval_state: str = "draft",
    ) -> ArtifactRecord:
        """Create or refresh a record, preserving history and approval state."""
        key = self._key(artifact_id, version)
        existing = self._records.get(key)
        now = _utcnow()
        if existing:
            if artifact_path:
                existing.artifact_path = artifact_path
            existing.updated_at = now
            self._save()
            return existing
        record = ArtifactRecord(
            artifact_id=artifact_id,
            version=version,
            artifact_path=artifact_path,
            approval_state=approval_state,  # type: ignore[arg-type]
            first_seen=now,
            updated_at=now,
        )
        self._records[key] = record
        self._save()
        return record

    def approve(
        self,
        artifact_id: str,
        version: str,
        by: str,
        notes: str = "",
        artifact_path: str = "",
    ) -> ArtifactRecord:
        """Move a capability to approved, recording reviewer and timestamp."""
        record = self.register(artifact_id, version, artifact_path)
        record.approval_state = "approved"
        record.approved_by = by
        record.approved_at = _utcnow()
        if notes:
            record.review_notes = (
                (record.review_notes + "\n" if record.review_notes else "")
                + f"Approved by {by}: {notes}"
            )
        record.updated_at = _utcnow()
        self._save()
        return record

    def reject(
        self,
        artifact_id: str,
        version: str,
        by: str,
        reason: str,
        artifact_path: str = "",
    ) -> ArtifactRecord:
        """Send a capability back to draft, recording why.

        Rejection is not deletion: the history stays, the approval fields are
        cleared, and the reason is appended to the review notes so the next
        reviewer sees what was wrong. The "deprecated" state is reserved for
        retiring a capability that was previously approved.
        """
        record = self.register(artifact_id, version, artifact_path)
        record.approval_state = "draft"
        record.approved_by = None
        record.approved_at = None
        record.review_notes = (
            (record.review_notes + "\n" if record.review_notes else "")
            + f"Rejected by {by}: {reason}"
        )
        record.updated_at = _utcnow()
        self._save()
        return record

    def record_replay(
        self,
        artifact_id: str,
        version: str,
        inputs: dict[str, Any],
        result_class: str,
    ) -> ArtifactRecord:
        """Append one replay outcome to the artifact's history."""
        record = self.register(artifact_id, version)
        record.replays.append(
            ReplayRecord(
                timestamp=_utcnow(),
                inputs_hash=hash_inputs(inputs or {}),
                result_class=result_class,  # type: ignore[arg-type]
            )
        )
        record.updated_at = _utcnow()
        self._save()
        return record

    def require_approved(self, artifact_id: str, version: str) -> ArtifactRecord:
        """Return the record if the capability is approved, else raise.

        The error message names the exact command that grants approval, so an
        operator or calling agent knows what to do next.
        """
        record = self.get(artifact_id, version)
        state = record.approval_state if record else "draft"
        if state == "approved":
            assert record is not None
            return record
        path_hint = ""
        if record and record.artifact_path:
            path_hint = f" --artifact {record.artifact_path}"
        raise ApprovalError(
            f"Artifact '{artifact_id}' v{version} is not approved for "
            f"unattended replay (state: {state}). Approve it first:\n"
            f"  python tools/approve.py approve{path_hint} --by <reviewer-name>"
        )
