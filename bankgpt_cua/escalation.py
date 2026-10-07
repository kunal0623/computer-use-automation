"""Human escalation and handoff for the BankGPT computer-use agent.

Control flow is a small state machine:

    AUTO -> PAUSED -> HUMAN -> AUTO

- ``request_intervention`` moves the run to PAUSED and records an
  ``InterventionRequest`` (optionally with a screenshot).
- ``takeover`` moves PAUSED to HUMAN: the operator takes the headed
  browser window and acts directly.
- ``handback`` moves HUMAN back to AUTO: the agent resumes.

While a human holds the session, ``inject_recorder`` captures their
browser events (click/input/change) into a buffer, and
``collect_human_actions`` drains it into the manager via
``record_human_action`` so the evidence log shows what the human did.

``create_operator_app`` builds a small FastAPI console the operator uses
to take over, hand back, and review recorded human actions. The console
does not drive the browser itself; the operator acts directly in the
headed browser window showing the live session.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Template
from pydantic import BaseModel, Field


class ControlState(str, Enum):
    AUTO = "AUTO"
    PAUSED = "PAUSED"
    HUMAN = "HUMAN"


class InterventionRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str
    goal: str
    step_index: int
    reason: str
    state_summary: str
    screenshot_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "open"


class EscalationManager:
    """Owns the control state machine and the human-action record."""

    def __init__(self, evidence_dir: str | None = None) -> None:
        self.evidence_dir = evidence_dir
        self._state = ControlState.AUTO
        self.interventions: list[InterventionRequest] = []
        self._human_actions: list[dict] = []

    @property
    def state(self) -> ControlState:
        return self._state

    @property
    def human_actions(self) -> list[dict]:
        return list(self._human_actions)

    def open_interventions(self) -> list[InterventionRequest]:
        return [i for i in self.interventions if i.status == "open"]

    def request_intervention(
        self,
        run_id: str,
        goal: str,
        step_index: int,
        reason: str,
        state_summary: str,
        screenshot: bytes | None = None,
    ) -> InterventionRequest:
        screenshot_path: str | None = None
        request_id = uuid.uuid4().hex
        if screenshot is not None and self.evidence_dir:
            target_dir = os.path.join(self.evidence_dir, "interventions")
            os.makedirs(target_dir, exist_ok=True)
            screenshot_path = os.path.join(target_dir, f"{request_id}.png")
            with open(screenshot_path, "wb") as handle:
                handle.write(screenshot)
        request = InterventionRequest(
            id=request_id,
            run_id=run_id,
            goal=goal,
            step_index=step_index,
            reason=reason,
            state_summary=state_summary,
            screenshot_path=screenshot_path,
        )
        self.interventions.append(request)
        self._state = ControlState.PAUSED
        return request

    def takeover(self) -> None:
        if self._state is not ControlState.PAUSED:
            raise RuntimeError(
                f"Cannot take over from state {self._state.value}: must be PAUSED"
            )
        self._state = ControlState.HUMAN
        for request in self.interventions:
            if request.status == "open":
                request.status = "taken_over"

    def handback(self) -> None:
        if self._state is not ControlState.HUMAN:
            raise RuntimeError(
                f"Cannot hand back from state {self._state.value}: must be HUMAN"
            )
        self._state = ControlState.AUTO
        for request in self.interventions:
            if request.status == "taken_over":
                request.status = "handed_back"

    def record_human_action(self, description: str, detail: dict | None = None) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "detail": detail or {},
        }
        self._human_actions.append(record)
        return record


_HUMAN_RECORDER_JS = """(() => {
  if (window.__bankgpt_recorder_installed) return;
  window.__bankgpt_recorder_installed = true;
  window.__bankgpt_human_actions = window.__bankgpt_human_actions || [];
  function describe(el) {
    if (!el || !el.tagName) return "";
    let s = el.tagName.toLowerCase();
    if (el.id) s += "#" + el.id;
    if (el.getAttribute && el.getAttribute("name")) s += "[name=" + el.getAttribute("name") + "]";
    return s;
  }
  function record(kind, el) {
    if (window.__bankgpt_control !== "HUMAN") return;
    let text = "";
    if (el && el.innerText) text = el.innerText.slice(0, 120);
    if (!text && el && el.value !== undefined) text = String(el.value).slice(0, 120);
    window.__bankgpt_human_actions.push({
      kind: kind,
      target: describe(el),
      text: text,
      ts: new Date().toISOString(),
    });
  }
  document.addEventListener("click", (e) => record("click", e.target), true);
  document.addEventListener("input", (e) => record("input", e.target), true);
  document.addEventListener("change", (e) => record("change", e.target), true);
})();"""


def inject_recorder(page: Any) -> None:
    """Install the human-action recorder init script on a Playwright page."""
    page.add_init_script(_HUMAN_RECORDER_JS)


def set_control(page: Any, state: ControlState) -> None:
    """Mirror the control state into the page so the recorder knows when to capture."""
    page.evaluate(f"window.__bankgpt_control = {state.value!r};")


def collect_human_actions(page: Any) -> list[dict]:
    """Read and clear the buffered human actions from the page."""
    return page.evaluate(
        """(() => {
      const actions = window.__bankgpt_human_actions || [];
      window.__bankgpt_human_actions = [];
      return actions;
    })()"""
    )


_OPERATOR_TEMPLATE = Template(
    """<!doctype html>
<html>
<head><meta charset="utf-8"><title>BankGPT Operator Console</title>
<style>
body { font-family: sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }
.state { font-size: 1.4em; font-weight: bold; }
.card { border: 1px solid #ccc; border-radius: 8px; padding: 1em; margin: 1em 0; }
button { font-size: 1em; padding: 0.5em 1em; margin-right: 0.5em; }
.note { color: #555; }
</style>
</head>
<body>
<h1>BankGPT Operator Console</h1>
<p class="state">Control state: {{ state }}</p>
<div class="card">
<h2>Session</h2>
<p>Goal: {{ session.goal }}</p>
<p>Run dir: {{ session.run_dir }}</p>
<p class="note">The operator acts directly in the headed browser window showing the live session. This console only switches control state and records what happened.</p>
<form method="post" action="/api/takeover"><button type="submit">Take over</button></form>
<form method="post" action="/api/handback"><button type="submit">Hand back to agent</button></form>
</div>
<div class="card">
<h2>Open interventions ({{ interventions|length }})</h2>
{% if interventions %}
{% for i in interventions %}
<p><b>{{ i.goal }}</b> at step {{ i.step_index }}<br>
Reason: {{ i.reason }}<br>
State: {{ i.state_summary }}</p>
{% endfor %}
{% else %}
<p class="note">None.</p>
{% endif %}
</div>
<div class="card">
<h2>Recorded human actions ({{ human_actions|length }})</h2>
{% if human_actions %}
<ul>
{% for a in human_actions %}
<li>{{ a.ts }}: {{ a.description }}</li>
{% endfor %}
</ul>
{% else %}
<p class="note">None recorded yet.</p>
{% endif %}
</div>
</body>
</html>"""
)


def create_operator_app(manager: EscalationManager, session_info: dict) -> FastAPI:
    """Build the operator console FastAPI app for one run's manager."""
    app = FastAPI(title="BankGPT Operator Console")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _OPERATOR_TEMPLATE.render(
            state=manager.state.value,
            session={
                "goal": session_info.get("goal", ""),
                "run_dir": session_info.get("run_dir", ""),
            },
            interventions=[i.model_dump() for i in manager.open_interventions()],
            human_actions=manager.human_actions,
        )

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        return JSONResponse(
            {
                "state": manager.state.value,
                "open_interventions": len(manager.open_interventions()),
            }
        )

    @app.post("/api/takeover")
    def api_takeover() -> JSONResponse:
        try:
            manager.takeover()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"state": manager.state.value})

    @app.post("/api/handback")
    def api_handback() -> JSONResponse:
        try:
            manager.handback()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"state": manager.state.value})

    @app.get("/api/human_actions")
    def api_human_actions() -> JSONResponse:
        return JSONResponse({"human_actions": manager.human_actions})

    return app
