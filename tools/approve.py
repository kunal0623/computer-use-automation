"""Manage capability approval state.

The approval lifecycle is draft -> approved (-> deprecated for retirement).
Only approved capabilities may run unattended; see tools/replay.py
--unattended and bankgpt_cua/catalog.py.

Usage:
    python tools/approve.py approve --artifact <path> --by <name> [--notes ...]
    python tools/approve.py reject  --artifact <path> --by <name> --notes ...
    python tools/approve.py status  --artifact <path>
    python tools/approve.py status  --id <artifact-id> --version <semver>

Approval is mirrored in two places: the registry (source of truth for
execution) and the artifact file itself (so the artifact stays
self-describing for reviewers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summarize(record) -> dict:
    reliability = record.reliability()
    return {
        "artifact_id": record.artifact_id,
        "version": record.version,
        "artifact_path": record.artifact_path,
        "approval_state": record.approval_state,
        "approved_by": record.approved_by,
        "approved_at": record.approved_at,
        "reliability": reliability,
        "replay_counts": record.counts(),
        "total_replays": len(record.replays),
        "review_notes": record.review_notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage capability approval.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_approve = sub.add_parser("approve", help="Approve a capability.")
    p_approve.add_argument("--artifact", required=True, help="Path to artifact.json.")
    p_approve.add_argument("--by", required=True, help="Reviewer name.")
    p_approve.add_argument("--notes", default="", help="Review notes.")

    p_reject = sub.add_parser("reject", help="Send a capability back to draft.")
    p_reject.add_argument("--artifact", required=True, help="Path to artifact.json.")
    p_reject.add_argument("--by", required=True, help="Reviewer name.")
    p_reject.add_argument("--notes", required=True, help="Why it was rejected.")

    p_status = sub.add_parser("status", help="Show approval state and score.")
    p_status.add_argument("--artifact", default=None, help="Path to artifact.json.")
    p_status.add_argument("--id", default=None, help="Artifact id (with --version).")
    p_status.add_argument("--version", default=None, help="Artifact version.")

    args = parser.parse_args()

    from bankgpt_cua.artifact import CapabilityArtifact
    from bankgpt_cua.registry import Registry

    registry = Registry()

    if args.command == "status":
        if args.artifact:
            artifact = CapabilityArtifact.load(args.artifact)
            artifact_id, version = artifact.id, artifact.version
        elif args.id and args.version:
            artifact_id, version = args.id, args.version
        else:
            print("error: status needs --artifact or --id with --version",
                  file=sys.stderr)
            return 1
        record = registry.get(artifact_id, version)
        if record is None:
            print(json.dumps({
                "artifact_id": artifact_id,
                "version": version,
                "approval_state": "draft",
                "reliability": None,
                "total_replays": 0,
                "note": "not yet registered; no replay history",
            }, indent=2))
            return 0
        print(json.dumps(_summarize(record), indent=2))
        return 0

    # approve / reject: load, patch the artifact file, then update registry.
    artifact = CapabilityArtifact.load(args.artifact)
    if args.command == "approve":
        artifact.approval_state = "approved"
        artifact.approved_by = args.by
        artifact.approved_at = _utcnow()
        if args.notes:
            artifact.review_notes = (
                (artifact.review_notes + "\n" if artifact.review_notes else "")
                + f"Approved by {args.by}: {args.notes}"
            )
        artifact.save(args.artifact)
        record = registry.approve(
            artifact.id, artifact.version, args.by, args.notes,
            artifact_path=args.artifact,
        )
        print(f"approved '{artifact.id}' v{artifact.version} by {args.by}")
    else:
        artifact.approval_state = "draft"
        artifact.approved_by = None
        artifact.approved_at = None
        artifact.review_notes = (
            (artifact.review_notes + "\n" if artifact.review_notes else "")
            + f"Rejected by {args.by}: {args.notes}"
        )
        artifact.save(args.artifact)
        record = registry.reject(
            artifact.id, artifact.version, args.by, args.notes,
            artifact_path=args.artifact,
        )
        print(f"rejected '{artifact.id}' v{artifact.version}; back to draft")

    print(json.dumps(_summarize(record), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
