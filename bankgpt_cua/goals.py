"""Background goal-run jobs for the Studio chat page.

A chat goal runs the real agent loop (``bankgpt_cua.agent.run_goal``) in a
background thread while the browser page streams per-step progress as
server-sent events. Jobs are in-memory only; this is demo tooling, not a
job queue.

Safety properties:
- Guardrails stay enforced by the agent loop itself; this module adds no
  path that bypasses them.
- At most one run at a time (the browser session is a shared resource);
  a second start is refused with a clear message.
- Artifact building reuses the shared ``bankgpt_cua.discovery`` helpers,
  so chat-built artifacts are identical in shape to CLI-built ones.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import threading
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from bankgpt_cua.agent import run_goal
from bankgpt_cua.discovery import (
    build_artifact,
    default_secret_values,
    DEFAULT_INPUTS,
    save_artifact_json,
)
from bankgpt_cua.escalation import EscalationManager
from bankgpt_cua.evidence import new_run_dir, RunLogger
from bankgpt_cua.guardrails import Policy, redact_dict
from bankgpt_cua.llm import default_demo_script, OpenAICompatClient, ScriptedMockClient
from bankgpt_cua.surface import WebSurface

DEFAULT_CHAT_ENTRY = "http://127.0.0.1:8765/login"
DEFAULT_MAX_STEPS = 25
DEFAULT_TIMEOUT_S = 600

TERMINAL_STATUSES = (
    "completed",
    "stuck",
    "escalated",
    "timeout",
    "guardrail_blocked",
    "error",
)

# Indirection points so tests can run jobs without a browser or an LLM.
RUN_GOAL_FN: Callable[..., Any] = run_goal
SURFACE_FACTORY: Callable[[bool], Any] | None = None


class GoalError(ValueError):
    """The goal request was invalid (bad input, missing key, mock down)."""


class BusyError(RuntimeError):
    """A goal run is already in progress."""


def llm_key_available() -> bool:
    """True when a real-LLM API key is configured in the environment."""
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@dataclass
class GoalJob:
    """In-memory state for one chat goal run."""

    id: str
    goal: str
    client: str
    headless: bool
    entry: str
    status: str = "running"
    events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)
    run_dir: str = ""
    artifact_id: str | None = None
    artifact_path: str | None = None
    error: str | None = None
    final: dict[str, Any] | None = None


_jobs: dict[str, GoalJob] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()


def get_job(job_id: str) -> GoalJob | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _describe_target_text(strategies: list[dict[str, Any]]) -> str:
    """Plain-text one-liner for a recorded step's locator strategies."""
    hints: list[str] = []
    for s in strategies or []:
        kind = s.get("type")
        if kind == "role":
            name = s.get("name")
            hints.append(
                f"{s.get('role') or 'element'} '{name}'" if name else (s.get("role") or "element")
            )
        elif kind in ("label", "placeholder", "text"):
            hints.append(f"{kind} '{s.get('value')}'")
        elif kind in ("css", "xpath"):
            hints.append(f"{kind} {(s.get('value') or '')[:60]}")
        elif kind == "attr":
            hints.append(f"@{s.get('attr_name')}='{s.get('attr_value')}'")
        elif kind == "image":
            hints.append("image template")
    return "; ".join(hints) if hints else "no target"


class _ChatLogger:
    """RunLogger wrapper that also streams per-step events to the job queue.

    Screenshots are captured from the live page after each recorded step
    and saved into the run's evidence directory, exactly like the CLI
    evidence pipeline. The redaction behavior is unchanged.
    """

    def __init__(self, inner: RunLogger, job: GoalJob, surface: Any) -> None:
        self._inner = inner
        self._job = job
        self._surface = surface

    def log(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self._inner.log(event, payload)
        if event == "step":
            step = payload.get("step", {}) or {}
            index = step.get("index", 0)
            shot_url = None
            try:
                page = getattr(self._surface, "page", None)
                if page is not None:
                    png = page.screenshot()
                    self._inner.screenshot(f"step-{index}", png)
                    shot_url = f"/api/goals/{self._job.id}/shots/{index}"
            except Exception:
                shot_url = None
            self._job.events.put(
                {
                    "event": "step",
                    "data": {
                        "index": index,
                        "action": step.get("action"),
                        "target": _describe_target_text(step.get("target_strategies") or []),
                        "reasoning": (step.get("reasoning") or "")[:240],
                        "shot": shot_url,
                    },
                }
            )
        return record

    def finalize(self, run_id: str) -> dict[str, Any]:
        return self._inner.finalize(run_id)

    def screenshot(self, name: str, png_bytes: bytes) -> str:
        return self._inner.screenshot(name, png_bytes)

    def dom(self, name: str, html: str) -> str:
        return self._inner.dom(name, html)


def _make_surface(headless: bool) -> Any:
    if SURFACE_FACTORY is not None:
        return SURFACE_FACTORY(headless)
    return WebSurface()


def _execute(job: GoalJob, *, evidence_base: str) -> None:
    """Worker body: run the goal, build the artifact, publish the final event."""
    secret_values = default_secret_values(DEFAULT_INPUTS)

    def _redact(payload: Any) -> Any:
        return redact_dict(payload, extra_values=secret_values)

    run_dir = new_run_dir(evidence_base, "chat-goal")
    job.run_dir = run_dir
    logger = RunLogger(run_dir, redact_fn=_redact)
    surface = _make_surface(job.headless)
    chat_logger = _ChatLogger(logger, job, surface)
    try:
        if job.client == "mock":
            llm: Any = ScriptedMockClient(default_demo_script(job.entry))
        else:
            llm = OpenAICompatClient()
        surface.start(job.entry, headless=job.headless)
        run = RUN_GOAL_FN(
            job.goal,
            job.entry,
            llm,
            surface,
            policy=Policy(),
            escalation_mgr=EscalationManager(evidence_dir=run_dir),
            logger=chat_logger,
            max_steps=DEFAULT_MAX_STEPS,
            timeout_s=DEFAULT_TIMEOUT_S,
        )
        status = getattr(run, "status", "unknown")
        run_id = getattr(run, "run_id", os.path.basename(run_dir))
        logger.finalize(run_id)
        artifact_id = None
        artifact_path = None
        if status == "completed":
            artifact = build_artifact(
                run,
                artifact_id=f"artifact-{run_id}",
                name=job.goal,
                review_notes="Built by the Studio chat page from a completed agent run.",
            )
            artifact_path = str(save_artifact_json(artifact, run_dir))
            artifact_id = artifact.id
        final: dict[str, Any] = {
            "status": status,
            "artifact_id": artifact_id,
            "artifact_path": artifact_path,
            "run_dir": run_dir,
        }
    except Exception as exc:  # never leave the client hanging
        final = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            surface.close()
        except Exception:
            pass
        _run_lock.release()
    job.status = final["status"]
    job.artifact_id = final.get("artifact_id")
    job.artifact_path = final.get("artifact_path")
    job.error = final.get("error")
    job.final = final
    job.events.put({"event": "final", "data": final})


def start_goal(
    goal: str,
    client: str,
    headless: bool,
    entry: str,
    *,
    evidence_base: str,
) -> GoalJob:
    """Validate and start a chat goal run in a background thread.

    Raises GoalError for invalid requests and BusyError when a run is
    already in progress. The browser check and the key check happen here,
    before any thread or browser is started.
    """
    goal = (goal or "").strip()
    if not goal:
        raise GoalError("Goal must not be empty.")
    if client not in ("mock", "openai"):
        raise GoalError(f"Unknown client {client!r}; expected 'mock' or 'openai'.")
    if client == "openai" and not llm_key_available():
        raise GoalError(
            "No LLM API key is set. Set the LLM_API_KEY environment variable "
            "(or OPENAI_API_KEY) before starting a real-LLM run, or use the "
            "mock client instead."
        )
    if not _run_lock.acquire(blocking=False):
        raise BusyError("A goal run is already in progress. Wait for it to finish.")

    job = GoalJob(
        id=uuid.uuid4().hex[:12],
        goal=goal,
        client=client,
        headless=headless,
        entry=entry,
    )
    try:
        with _jobs_lock:
            _jobs[job.id] = job
        thread = threading.Thread(
            target=_execute, args=(job,), kwargs={"evidence_base": evidence_base}, daemon=True
        )
        thread.start()
    except Exception:
        _run_lock.release()
        with _jobs_lock:
            _jobs.pop(job.id, None)
        raise
    return job


def slugify(text: str) -> str:
    """Filename-safe slug for a capability name."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "capability"


def save_as_capability(job: GoalJob, capabilities_dir: str) -> dict[str, str]:
    """Copy a completed job's artifact into the capabilities directory.

    Returns the capability name, its studio URL, and the file path.
    Name collisions are resolved by suffixing (-2, -3, ...).
    """
    if job.status != "completed" or not job.artifact_path:
        raise GoalError(
            "Only a completed run with a built artifact can be saved as a capability."
        )
    src = job.artifact_path
    if not os.path.isfile(src):
        raise GoalError("The run's artifact file is missing; it cannot be saved.")
    import json as _json

    with open(src, encoding="utf-8") as handle:
        name = _json.load(handle).get("name") or job.goal
    os.makedirs(capabilities_dir, exist_ok=True)
    slug = slugify(name)
    dest = os.path.join(capabilities_dir, f"{slug}.json")
    counter = 2
    while os.path.exists(dest):
        dest = os.path.join(capabilities_dir, f"{slug}-{counter}.json")
        counter += 1
    shutil.copy(src, dest)
    return {
        "name": name,
        "url": f"/capabilities/{urllib.parse.quote(name, safe='')}",
        "path": dest,
    }
