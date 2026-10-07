"""Agent-facing capability catalog.

The catalog is the answer to "how does an AI agent actually use these
capabilities in production": it scans a directory of saved artifact files,
presents them as a typed, named tool surface (list with input schemas,
approval state, and reliability scores), and invokes them by name through
the deterministic replay engine. Invocation is unattended by definition, so
it carries the same approval gate as tools/replay.py --unattended: only
approved capabilities run.

Every invocation writes an evidence directory (manifest, result, and a note
recording that the call went through the catalog with zero LLM calls), so
catalog use stays auditable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bankgpt_cua.artifact import CapabilityArtifact
from bankgpt_cua.registry import ApprovalError, Registry, hash_inputs


class CatalogError(ValueError):
    """The catalog could not list, resolve, or invoke a capability."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_capabilities_dir() -> Path:
    """Capability directory; override with BANKGPT_CAPABILITIES."""
    return Path(os.environ.get("BANKGPT_CAPABILITIES", "./capabilities"))


class Catalog:
    """Discover and invoke saved capability artifacts by name."""

    def __init__(
        self,
        capabilities_dir: str | Path | None = None,
        registry: Registry | None = None,
        evidence_base: str = "./evidence",
    ) -> None:
        self.capabilities_dir = (
            Path(capabilities_dir)
            if capabilities_dir is not None
            else default_capabilities_dir()
        )
        self.registry = registry if registry is not None else Registry()
        self.evidence_base = evidence_base

    def _scan(self) -> list[tuple[Path, CapabilityArtifact]]:
        """Load every artifact JSON in the capabilities directory.

        Files that fail to parse are skipped, never fatal: a catalog must
        stay usable when one entry is bad.
        """
        found: list[tuple[Path, CapabilityArtifact]] = []
        if not self.capabilities_dir.is_dir():
            return found
        for path in sorted(self.capabilities_dir.glob("*.json")):
            try:
                found.append((path, CapabilityArtifact.load(path)))
            except Exception:
                continue
        return found

    def _approval_state(self, artifact: CapabilityArtifact) -> str:
        """Registry is the source of truth; fall back to the artifact file."""
        record = self.registry.get(artifact.id, artifact.version)
        if record is not None:
            return record.approval_state
        return artifact.approval_state

    def list_capabilities(self) -> list[dict[str, Any]]:
        """Describe every capability: contract, approval, reliability."""
        out: list[dict[str, Any]] = []
        for path, artifact in self._scan():
            record = self.registry.get(artifact.id, artifact.version)
            out.append(
                {
                    "name": artifact.name,
                    "id": artifact.id,
                    "version": artifact.version,
                    "description": artifact.description,
                    "inputs": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "required": p.required,
                            "description": p.description,
                        }
                        for p in artifact.inputs
                    ],
                    "outputs": [o.name for o in artifact.outputs],
                    "approval_state": self._approval_state(artifact),
                    "reliability": record.reliability() if record else None,
                    "total_replays": len(record.replays) if record else 0,
                    "path": str(path),
                }
            )
        return out

    def get(self, name: str) -> tuple[Path, CapabilityArtifact]:
        """Resolve a capability by name (or id). Errors on missing/ambiguous."""
        matches = [
            (p, a)
            for p, a in self._scan()
            if a.name == name or a.id == name
        ]
        if not matches:
            known = [a.name for _, a in self._scan()]
            raise CatalogError(
                f"Unknown capability {name!r}. Known: {known or '(none)'}"
            )
        if len(matches) > 1:
            ids = [f"{a.id}@{a.version}" for _, a in matches]
            raise CatalogError(
                f"Ambiguous capability name {name!r}; matches {ids}. "
                "Invoke by id instead."
            )
        return matches[0]

    def invoke(
        self, name: str, inputs: dict[str, Any], *, headless: bool = True
    ):
        """Invoke a capability by name through the deterministic replay engine.

        Steps: resolve -> approval gate -> input validation -> replay ->
        record -> evidence. Raises CatalogError for unknown capabilities,
        unapproved capabilities, and invalid inputs, all before any browser
        work starts. Returns the replay engine's structured ReplayResult.
        """
        from bankgpt_cua.replay import _validate_inputs, replay  # shared validation

        path, artifact = self.get(name)

        try:
            self.registry.require_approved(artifact.id, artifact.version)
        except ApprovalError as exc:
            raise CatalogError(str(exc)) from exc

        _, violations = _validate_inputs(artifact, inputs or {})
        if violations:
            raise CatalogError(
                f"Invalid inputs for capability {artifact.name!r}: "
                + "; ".join(violations)
            )

        from bankgpt_cua.surface import WebSurface
        from bankgpt_cua.guardrails import Policy, redact_dict
        from bankgpt_cua.escalation import EscalationManager
        from bankgpt_cua.evidence import RunLogger, new_run_dir

        os.makedirs(self.evidence_base, exist_ok=True)
        run_dir = new_run_dir(self.evidence_base, "catalog-invoke")
        secret_values = [
            str(inputs[p.name])
            for p in artifact.inputs
            if p.redact_in_logs and p.name in inputs and inputs[p.name] is not None
        ]

        def _redact(payload):
            return redact_dict(payload, extra_values=secret_values)

        logger = RunLogger(run_dir, redact_fn=_redact)
        try:
            surface = WebSurface(headless=headless)
        except TypeError:
            surface = WebSurface()

        result = replay(
            artifact,
            inputs,
            surface,
            policy=Policy(),
            escalation_mgr=EscalationManager(evidence_dir=run_dir),
            logger=logger,
            headless=headless,
        )
        logger.finalize(result.run_id)

        status = result.status
        if status in ("success", "business_outcome", "recovered", "hard_failure"):
            try:
                self.registry.record_replay(
                    artifact.id, artifact.version, inputs, status
                )
            except Exception:
                pass  # the registry must never break a run

        manifest = {
            "capability_name": artifact.name,
            "artifact_id": artifact.id,
            "artifact_version": artifact.version,
            "invoked_at": _utcnow(),
            "inputs_hash": hash_inputs(inputs or {}),
            "approval_state_at_invoke": self._approval_state(artifact),
            "model": None,
            "run_dir": run_dir,
        }
        Path(run_dir, "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        Path(run_dir, "result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        Path(run_dir, "NOTE.txt").write_text(
            "This invocation went through the agent-facing capability catalog "
            "(bankgpt_cua/catalog.py). The capability was invoked by name with "
            "typed inputs; execution used the deterministic replay engine "
            "with zero LLM calls. See manifest.json and result.json.\n",
            encoding="utf-8",
        )
        return result
