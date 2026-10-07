"""Versioned capability artifact schema for the BankGPT computer-use agent.

A CapabilityArtifact is the reviewed, deterministic record of a capability:
what it does, what inputs it takes, the exact steps to replay it, what it
extracts, how it verifies success, and how known error states are handled.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from bankgpt_cua.agent import AgentRun

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class LocatorStrategy(BaseModel):
    """One way to find a UI element; strategies are tried in order."""

    type: Literal["css", "xpath", "text", "role", "placeholder", "label", "attr", "image"]
    value: str | None = None
    role: str | None = None
    name: str | None = None
    attr_name: str | None = None
    attr_value: str | None = None
    template: str | None = None


class LocatorTarget(BaseModel):
    """Ordered chain of locator strategies for one target element."""

    strategies: list[LocatorStrategy] = Field(min_length=1)
    rationale: str = Field(description="Why this chain survives UI churn.")


class Checkpoint(BaseModel):
    """A verifiable post-condition for a step or a whole capability."""

    type: Literal["url_contains", "text_present", "element_visible"]
    value: str
    description: str


class ArtifactStep(BaseModel):
    """One deterministic replay step."""

    index: int = Field(ge=0)
    action: Literal["navigate", "click", "fill", "press", "select", "wait", "read", "done"]
    target: LocatorTarget | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint: Checkpoint | None = None
    notes: str = ""


class InputParam(BaseModel):
    """One declared input of a capability."""

    name: str
    type: Literal["string", "integer", "number", "boolean"]
    required: bool = True
    description: str = ""
    pattern: str | None = None
    example: str | None = None
    redact_in_logs: bool = False


class OutputDef(BaseModel):
    """One value extracted from the UI during replay."""

    name: str
    type: str
    description: str = ""
    extraction: dict[str, Any] = Field(
        description="Keys: step_index (int), strategy (a LocatorStrategy as dict), "
        "postprocess ('none' | 'strip_currency' | 'extract_currency' | 'strip_whitespace')."
    )


class ErrorPolicyEntry(BaseModel):
    """How replay should react to one known error state."""

    name: str
    outcome_class: Literal["business_outcome", "recoverable", "hard_failure"]
    match: dict[str, Any] = Field(
        description="Keys text_contains / url_contains / selector_present, str or list."
    )
    outcome: dict[str, Any] = Field(description="{code: str, message_template: str}")
    recovery: dict[str, Any] | None = Field(
        default=None,
        description="An action dict, e.g. {'action': 'click', 'target': {'strategies': [...]}} "
        "or {'action': 'wait', 'params': {'ms': 2000}} "
        "or {'action': 'navigate', 'params': {'url': '...'}}.",
    )


class CapabilityArtifact(BaseModel):
    """The full versioned capability record."""

    schema_version: Literal["1.0", "2.0"] = "2.0"
    id: str
    name: str
    version: str = Field(description="Semver, e.g. '1.0.0'.")
    description: str
    surface: dict[str, Any] = Field(description="{kind: 'web', entry_point: str, notes: str}")
    inputs: list[InputParam] = Field(default_factory=list)
    outputs: list[OutputDef] = Field(default_factory=list)
    steps: list[ArtifactStep] = Field(default_factory=list)
    success_condition: Checkpoint
    error_policy: list[ErrorPolicyEntry] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(
        description="{run_id, model, recorded_at, goal}"
    )
    review_notes: str = ""
    approval_state: Literal["draft", "approved", "deprecated"] = Field(
        default="draft",
        description="Lifecycle state. New artifacts start as draft; only "
        "approved capabilities may run unattended.",
    )
    approved_by: str | None = Field(
        default=None, description="Reviewer who approved, if any."
    )
    approved_at: str | None = Field(
        default=None, description="ISO UTC timestamp of approval, if any."
    )

    @field_validator("version")
    @classmethod
    def _version_is_semver(cls, v: str) -> str:
        if not SEMVER_RE.match(v):
            raise ValueError(f"version must be semver (X.Y.Z), got {v!r}")
        return v

    @model_validator(mode="after")
    def _step_indices_unique(self) -> "CapabilityArtifact":
        indices = [s.index for s in self.steps]
        if len(set(indices)) != len(indices):
            raise ValueError(f"step indices must be unique, got {indices}")
        return self

    def save(self, path: str | Path) -> Path:
        """Write the artifact to disk as pretty JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CapabilityArtifact":
        """Load and validate an artifact from disk."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON schema of the artifact (for review tooling)."""
        return cls.model_json_schema()

    def to_summary(self) -> str:
        """Human-readable review text. Plain prose, no em dashes."""
        lines: list[str] = []
        lines.append(f"{self.name} (v{self.version})")
        lines.append(f"ID: {self.id} | schema: {self.schema_version}")
        lines.append(f"Approval: {self.approval_state}" + (
            f" by {self.approved_by} at {self.approved_at}"
            if self.approval_state == "approved" and self.approved_by
            else ""
        ))
        lines.append(self.description)
        lines.append("")
        lines.append("Inputs:")
        if not self.inputs:
            lines.append("  (none)")
        for inp in self.inputs:
            req = "required" if inp.required else "optional"
            pat = f", pattern={inp.pattern}" if inp.pattern else ""
            lines.append(f"  - {inp.name}: {inp.type} ({req}{pat})")
            if inp.description:
                lines.append(f"      {inp.description}")
        lines.append("")
        lines.append("Outputs:")
        if not self.outputs:
            lines.append("  (none)")
        for out in self.outputs:
            ex = out.extraction or {}
            lines.append(
                f"  - {out.name}: {out.type} "
                f"(step {ex.get('step_index')}, postprocess={ex.get('postprocess', 'none')})"
            )
            if out.description:
                lines.append(f"      {out.description}")
        lines.append("")
        lines.append("Steps:")
        for step in sorted(self.steps, key=lambda s: s.index):
            lines.append(f"  {step.index}. {step.action}")
            if step.target:
                strat = ", ".join(
                    s.type + (f"={s.value or s.name or s.role or ''}") for s in step.target.strategies
                )
                lines.append(f"      target: {strat}")
                lines.append(f"      rationale: {step.target.rationale}")
            if step.params:
                lines.append(f"      params: {step.params}")
            if step.checkpoint:
                cp = step.checkpoint
                lines.append(
                    f"      checkpoint [{cp.type}]: {cp.value!r} ({cp.description})"
                )
            if step.notes:
                lines.append(f"      notes: {step.notes}")
        lines.append("")
        sc = self.success_condition
        lines.append(
            f"Success condition [{sc.type}]: {sc.value!r} ({sc.description})"
        )
        lines.append("")
        lines.append("Error policy:")
        if not self.error_policy:
            lines.append("  (none)")
        for entry in self.error_policy:
            outcome = entry.outcome or {}
            lines.append(
                f"  - {entry.name}: {entry.outcome_class} "
                f"(code={outcome.get('code')})"
            )
            lines.append(f"      message: {outcome.get('message_template')}")
            if entry.recovery:
                lines.append(f"      recovery: {entry.recovery}")
        if self.review_notes:
            lines.append("")
            lines.append(f"Review notes: {self.review_notes}")
        return "\n".join(lines)


class ArtifactBuilder:
    """Builds a CapabilityArtifact from a recorded agent run."""

    @staticmethod
    def from_run(
        run: "AgentRun",
        *,
        artifact_id: str,
        name: str,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        error_policy: list[dict[str, Any]],
        review_notes: str = "",
    ) -> CapabilityArtifact:
        """Map each recorded step of an agent run to an artifact step.

        Raises ValueError when the run has no usable final state to derive
        the success condition from.
        """
        steps: list[ArtifactStep] = []
        for rs in getattr(run, "steps", []) or []:
            target = None
            strategies = getattr(rs, "target_strategies", None)
            if strategies:
                target = LocatorTarget(
                    strategies=[LocatorStrategy(**s) for s in strategies],
                    rationale=getattr(rs, "target_rationale", "") or "",
                )
            checkpoint = None
            raw_cp = getattr(rs, "checkpoint", None)
            if raw_cp:
                checkpoint = Checkpoint(**raw_cp) if isinstance(raw_cp, dict) else raw_cp
            reasoning = getattr(rs, "reasoning", "") or ""
            steps.append(
                ArtifactStep(
                    index=getattr(rs, "index", len(steps)),
                    action=getattr(rs, "action", "wait"),
                    target=target,
                    params=dict(getattr(rs, "params", {}) or {}),
                    checkpoint=checkpoint,
                    notes=reasoning[:500],
                )
            )

        final_text = ArtifactBuilder._final_observed_text(run)
        if not final_text:
            raise ValueError(
                "Run has no usable final state: no observed text found in any "
                "step's obs_after, so no success condition can be derived."
            )
        success_condition = Checkpoint(
            type="text_present",
            value=final_text,
            description=(
                f"Final recorded state from run {getattr(run, 'run_id', '?')}: "
                "expected text present after the last step."
            ),
        )

        recorded_at = getattr(run, "ended_at", None) or getattr(run, "started_at", None) or ""
        return CapabilityArtifact(
            schema_version="2.0",
            id=artifact_id,
            name=name,
            version="1.0.0",
            description=getattr(run, "goal", "") or "",
            surface={
                "kind": "web",
                "entry_point": getattr(run, "entry_point", "") or "",
                "notes": "",
            },
            inputs=[InputParam(**i) for i in inputs],
            outputs=[OutputDef(**o) for o in outputs],
            steps=steps,
            success_condition=success_condition,
            error_policy=[ErrorPolicyEntry(**e) for e in error_policy],
            provenance={
                "run_id": getattr(run, "run_id", "") or "",
                "model": getattr(run, "model_name", "") or "",
                "recorded_at": str(recorded_at),
                "goal": getattr(run, "goal", "") or "",
            },
            review_notes=review_notes,
        )

    @staticmethod
    def _final_observed_text(run: "AgentRun") -> str:
        """Find the last meaningful observed text in the run.

        Scans steps in reverse, preferring an explicit text field, then the
        page title, from each step's obs_after dict.
        """
        for rs in reversed(getattr(run, "steps", []) or []):
            obs = getattr(rs, "obs_after", None) or {}
            if not isinstance(obs, dict):
                continue
            text = (obs.get("text") or "").strip()
            if text:
                first_line = next(
                    (ln.strip() for ln in text.splitlines() if ln.strip()), ""
                )
                if first_line:
                    return first_line[:120]
            title = (obs.get("title") or "").strip()
            if title:
                return title[:120]
        return ""
