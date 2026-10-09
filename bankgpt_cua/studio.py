"""Macro Studio: a small web UI for understanding and demoing capabilities.

The Studio is a thin reading and execution layer over the capability
catalog. It renders saved artifacts in human terms (what the capability
does, what it needs, the steps it takes, how it handles errors) and lets
an operator replay a capability from a form, watching the structured
result come back. It is a demo and comprehension aid, not part of the
graded core, so it deliberately stays thin: server-rendered pages, no
frontend framework, and every run goes through the same deterministic
replay engine the CLI uses.

Safety properties inherited from the rest of the system:
- Input validation happens before any browser work starts.
- Unattended replays require an approved capability, with the same refusal
  message the CLI produces.
- Values of inputs flagged ``redact_in_logs`` are never rendered back to
  the page; history shows only the registry's input hashes.
"""

from __future__ import annotations

import json
import os
import queue
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from jinja2 import Template

from bankgpt_cua import goals
from bankgpt_cua.artifact import ArtifactStep, CapabilityArtifact, LocatorTarget
from bankgpt_cua.catalog import Catalog, CatalogError
from bankgpt_cua.registry import ApprovalError, Registry

STATUS_COLORS = {
    "success": "#1a7f37",
    "business_outcome": "#8250df",
    "recovered": "#9a6700",
    "hard_failure": "#cf222e",
}

APPROVAL_COLORS = {
    "draft": "#9a6700",
    "approved": "#1a7f37",
    "deprecated": "#57606a",
}

_BASE_CSS = """
body { font-family: sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #1f2328; }
a { color: #0969da; }
.card { border: 1px solid #d0d7de; border-radius: 8px; padding: 1em; margin: 1em 0; }
.badge { display: inline-block; padding: 0.15em 0.6em; border-radius: 999px; color: #fff; font-size: 0.85em; font-weight: bold; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #d0d7de; padding: 0.4em 0.6em; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }
.muted { color: #57606a; }
.warn { background: #fff8c5; border: 1px solid #d4a72c; border-radius: 8px; padding: 1em; margin: 1em 0; }
.error { background: #ffebe9; border: 1px solid #cf222e; border-radius: 8px; padding: 1em; margin: 1em 0; }
.ok { background: #dafbe1; border: 1px solid #1a7f37; border-radius: 8px; padding: 1em; margin: 1em 0; }
code { background: #f6f8fa; padding: 0.1em 0.3em; border-radius: 4px; }
pre { background: #f6f8fa; padding: 1em; border-radius: 8px; overflow-x: auto; }
input[type=text], input[type=number] { width: 100%; max-width: 420px; padding: 0.4em; font-size: 1em; }
select { font-size: 1em; padding: 0.4em; max-width: 420px; }
button { font-size: 1em; padding: 0.5em 1.2em; margin-top: 0.8em; cursor: pointer; }
.chat-step img { max-width: 100%; border: 1px solid #d0d7de; border-radius: 4px; margin-top: 0.5em; }
label { display: block; margin: 0.6em 0 0.2em; font-weight: bold; }
.step { border-left: 3px solid #0969da; padding-left: 0.8em; margin: 0.8em 0; }
.rationale { color: #57606a; font-style: italic; }
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_BASE_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def _approval_badge(state: str) -> str:
    color = APPROVAL_COLORS.get(state, "#57606a")
    return f"<span class='badge' style='background:{color}'>{state}</span>"


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#57606a")
    return f"<span class='badge' style='background:{color}'>{status}</span>"


def _reliability_text(record: Any) -> str:
    if record is None:
        return "<span class='muted'>no registry record</span>"
    score = record.reliability()
    if score is None:
        return "<span class='muted'>no replays yet</span>"
    return f"{score:.0%} ({len(record.replays)} replays)"


def _quote(name: str) -> str:
    return urllib.parse.quote(name, safe="")


def _describe_target(target: LocatorTarget | None) -> str:
    """Human-readable one-liner for a locator target, e.g. button "Close"."""
    if target is None:
        return "<span class='muted'>no target</span>"
    hints: list[str] = []
    for s in target.strategies:
        if s.type == "role":
            hints.append(f"{s.role or 'element'} {s.name!r}" if s.name else (s.role or "element"))
        elif s.type in ("label", "placeholder", "text"):
            hints.append(f"{s.type} {s.value!r}")
        elif s.type in ("css", "xpath"):
            hints.append(f"{s.type} <code>{(s.value or '')[:60]}</code>")
        elif s.type == "attr":
            hints.append(f"attr {s.attr_name}={s.attr_value!r}")
        elif s.type == "image":
            hints.append("image template")
    chain = " &rarr; ".join(s.type for s in target.strategies)
    return f"{'; '.join(hints)} <span class='muted'>(tries: {chain})</span>"


def _describe_step(step: ArtifactStep, redacted_names: set[str]) -> str:
    """Human-readable one-liner for a replay step."""
    text = f"<b>#{step.index} {step.action}</b>"
    if step.target:
        text += f" on {_describe_target(step.target)}"
    if step.params:
        params = {
            k: ("***" if k in redacted_names else v)
            for k, v in step.params.items()
        }
        text += f" with <code>{params}</code>"
    return text


def _check_entry_point(url: str, timeout: float = 3.0) -> bool:
    """True when the mock bank answers. Never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _masked_inputs(artifact: CapabilityArtifact, inputs: dict[str, Any]) -> dict[str, Any]:
    """Inputs for display: values flagged redact_in_logs are masked."""
    redacted = {p.name for p in artifact.inputs if p.redact_in_logs}
    return {k: ("***" if k in redacted else v) for k, v in inputs.items()}


_LIST_TEMPLATE = Template(
    """
<h1>Macro Studio</h1>
<p class="muted">Saved capability artifacts, rendered for humans. Pick one to
inspect it, replay it, or approve it.</p>
{% if not capabilities %}
<div class="card"><p class="muted">No capabilities found in
<code>{{ capabilities_dir }}</code>. Record one with
<code>tools/discover.py</code> and copy its <code>artifact.json</code>
here.</p></div>
{% endif %}
{% for cap in capabilities %}
<div class="card">
<h2><a href="/capabilities/{{ cap.url_name }}">{{ cap.name }}</a>
{{ cap.approval_badge | safe }}</h2>
<p>{{ cap.description }}</p>
<p class="muted">v{{ cap.version }} &middot; reliability:
{{ cap.reliability }} &middot; outputs: {{ cap.outputs|join(', ') }}</p>
</div>
{% endfor %}
<p><a href="/chat">Goal chat</a> &middot; <a href="/history">Replay history</a></p>
""",
    autoescape=True,
)


_DETAIL_TEMPLATE = Template(
    """
<p><a href="/">&larr; all capabilities</a></p>
<h1>{{ artifact.name }} {{ approval_badge | safe }}</h1>
<p>{{ artifact.description }}</p>
<p class="muted">ID <code>{{ artifact.id }}</code> &middot; v{{ artifact.version }}
&middot; schema {{ artifact.schema_version }} &middot; reliability:
{{ reliability }}</p>
{% if approval_state == 'approved' %}
<p class="muted">Approved by {{ artifact.approved_by }} at {{ artifact.approved_at }}</p>
{% endif %}

{% if unreachable_banner %}
<div class="error"><b>Mock bank is not reachable</b> at
<code>{{ entry_point }}</code>. Start it first:<br>
<code>python tools/serve_mock.py</code><br>
then reload this page to replay.</div>
{% endif %}

<div class="card">
<h2>Replay</h2>
<form method="post" action="/capabilities/{{ url_name }}/replay">
{% for inp in inputs %}
<label>{{ inp.name }} <span class="muted">({{ inp.type }}{% if inp.required %}, required{% else %}, optional{% endif %}{% if inp.pattern %}, pattern {{ inp.pattern }}{% endif %}{% if inp.redacted %} &middot; redacted in logs{% endif %})</span></label>
<input type="text" name="input:{{ inp.name }}" placeholder="{{ inp.example or '' }}">
{% if inp.description %}<div class="muted">{{ inp.description }}</div>{% endif %}
{% endfor %}
<label><input type="checkbox" name="headless" checked> Headless browser</label>
<label><input type="checkbox" name="unattended"> Unattended (requires approval)</label>
<button type="submit">Run replay</button>
</form>
<p class="muted">Replay runs the deterministic engine with zero LLM calls.
Unattended runs are refused unless the capability is approved.</p>
</div>

{% if approval_state == 'draft' %}
<div class="card">
<h2>Approve this capability</h2>
<form method="post" action="/capabilities/{{ url_name }}/approve">
<label>Reviewer name</label>
<input type="text" name="reviewer" placeholder="studio-operator">
<label>Notes (optional)</label>
<input type="text" name="notes" placeholder="why this is safe to run unattended">
<button type="submit">Approve</button>
</form>
</div>
{% endif %}

<div class="card">
<h2>Inputs</h2>
<table><tr><th>Name</th><th>Type</th><th>Required</th><th>Details</th></tr>
{% for inp in inputs %}
<tr><td><code>{{ inp.name }}</code></td><td>{{ inp.type }}</td>
<td>{{ 'yes' if inp.required else 'no' }}</td>
<td>{{ inp.description }}{% if inp.pattern %}<br>pattern: <code>{{ inp.pattern }}</code>{% endif %}{% if inp.redacted %}<br><span class="muted">redacted in logs</span>{% endif %}</td></tr>
{% endfor %}</table>
</div>

<div class="card">
<h2>Outputs</h2>
<table><tr><th>Name</th><th>Type</th><th>Extraction</th></tr>
{% for out in outputs %}
<tr><td><code>{{ out.name }}</code></td><td>{{ out.type }}</td>
<td>{{ out.description }}<br><span class="muted">step {{ out.step_index }}, postprocess: {{ out.postprocess }}</span></td></tr>
{% endfor %}</table>
</div>

<div class="card">
<h2>Steps ({{ steps|length }})</h2>
{% for step in steps %}
<div class="step">{{ step.html | safe }}<br>
<span class="rationale">Why this survives UI churn: {{ step.rationale }}</span>
{% if step.checkpoint %}<br><span class="muted">Checkpoint [{{ step.checkpoint_type }}]:
<code>{{ step.checkpoint_value }}</code> ({{ step.checkpoint_desc }})</span>{% endif %}
{% if step.notes %}<br><span class="muted">Notes: {{ step.notes }}</span>{% endif %}
</div>
{% endfor %}
</div>

<div class="card">
<h2>Success condition</h2>
<p><code>[{{ success.type }}] {{ success.value }}</code><br>
<span class="muted">{{ success.description }}</span></p>
</div>

<div class="card">
<h2>Error policy</h2>
<table><tr><th>Observed state</th><th>Outcome class</th><th>Code</th><th>Message</th><th>Recovery</th></tr>
{% for e in error_policy %}
<tr><td><b>{{ e.name }}</b><br><span class="muted">{{ e.match }}</span></td>
<td>{{ e.outcome_class }}</td><td><code>{{ e.code }}</code></td>
<td>{{ e.message }}</td><td>{{ e.recovery or '&mdash;' }}</td></tr>
{% endfor %}</table>
</div>

<div class="card">
<h2>Provenance</h2>
<p class="muted">Recorded from run <code>{{ provenance.run_id }}</code> by model
<code>{{ provenance.model }}</code> at {{ provenance.recorded_at }}.<br>
Goal: {{ provenance.goal }}</p>
{% if artifact.review_notes %}<p>Review notes: {{ artifact.review_notes }}</p>{% endif %}
</div>

<div class="card">
<h2>Recent replays</h2>
{% if history %}
<table><tr><th>When (UTC)</th><th>Result</th><th>Inputs</th></tr>
{% for h in history %}
<tr><td>{{ h.timestamp }}</td><td>{{ h.badge | safe }}</td>
<td><code>{{ h.inputs_hash }}</code> <span class="muted">(hash only)</span></td></tr>
{% endfor %}</table>
{% else %}
<p class="muted">No replays recorded yet.</p>
{% endif %}
</div>
""",
    autoescape=True,
)


_RESULT_TEMPLATE = Template(
    """
<p><a href="/capabilities/{{ url_name }}">&larr; back to {{ name }}</a></p>
<h1>Replay result {{ status_badge | safe }}</h1>
{% if warning %}
<div class="warn">{{ warning }}</div>
{% endif %}
<p class="muted">Inputs used: <code>{{ inputs }}</code></p>
{% if outputs %}
<div class="card"><h2>Outputs</h2>
<table><tr><th>Name</th><th>Value</th></tr>
{% for k, v in outputs.items() %}
<tr><td><code>{{ k }}</code></td><td>{{ v }}</td></tr>
{% endfor %}</table></div>
{% endif %}
{% if business_outcome %}
<div class="card"><h2>Business outcome</h2>
<p><code>{{ business_outcome.code }}</code>: {{ business_outcome.message }}</p></div>
{% endif %}
{% if recoveries %}
<div class="card"><h2>Recoveries applied</h2>
<ul>{% for r in recoveries %}<li>{{ r }}</li>{% endfor %}</ul></div>
{% endif %}
{% if failure %}
<div class="error"><h2>Hard failure</h2>
<p>Step {{ failure.step_index }}: expected <code>{{ failure.expected }}</code>,
observed <code>{{ failure.observed }}</code>.</p>
<p class="muted">Screenshot and DOM snapshot saved to the run directory.</p></div>
{% endif %}
<p class="muted">Run directory: <code>{{ run_dir }}</code></p>
""",
    autoescape=True,
)


_HISTORY_TEMPLATE = Template(
    """
<p><a href="/">&larr; all capabilities</a></p>
<h1>Replay history</h1>
<p class="muted">From the local registry. Input values are never stored, only
hashes, so history shows result classes, not data.</p>
{% for cap in capabilities %}
<div class="card">
<h2><a href="/capabilities/{{ cap.url_name }}">{{ cap.name }}</a>
{{ cap.approval_badge | safe }}</h2>
{% if cap.history %}
<table><tr><th>When (UTC)</th><th>Result</th><th>Inputs</th></tr>
{% for h in cap.history %}
<tr><td>{{ h.timestamp }}</td><td>{{ h.badge | safe }}</td>
<td><code>{{ h.inputs_hash }}</code></td></tr>
{% endfor %}</table>
{% else %}
<p class="muted">No replays recorded yet.</p>
{% endif %}
</div>
{% endfor %}
""",
    autoescape=True,
)


_CHAT_PAGE = """
<p><a href="/">&larr; all capabilities</a></p>
<h1>Goal chat</h1>
<p class="muted">Type a goal in plain language. The agent drives the mock bank
live in a browser, and each step lands here as it happens: the action, what it
targeted, the model's reasoning, and a screenshot.</p>
<div class="card">
<label for="goal">Goal</label>
<input type="text" id="goal" style="max-width:640px"
placeholder="Look up member 12345 and read their current savings balance">
<label for="client">Client</label>
<select id="client">
<option value="mock">mock: scripted client, deterministic, no API key needed</option>
<option value="openai">openai: real LLM, needs LLM_API_KEY set</option>
</select>
<div id="keywarn" class="warn" style="display:none"><b>No LLM API key detected.</b>
Set the <code>LLM_API_KEY</code> environment variable before starting a real-LLM
run, or use the mock client.</div>
<label><input type="checkbox" id="headless" checked> Headless browser
<span class="muted">(uncheck to watch the browser window)</span></label>
<br><button id="run">Run goal</button>
<p class="muted">One run at a time: the browser session is a shared resource.</p>
</div>
<div id="chat"></div>
<script>
(function () {
  var chat = document.getElementById('chat');
  var running = false;

  function scrollDown(el) { el.scrollIntoView(false); }

  function note(text, cls) {
    var d = document.createElement('div');
    d.className = cls || 'card';
    d.textContent = text;
    chat.appendChild(d);
    scrollDown(d);
    return d;
  }

  function addStep(d) {
    var card = document.createElement('div');
    card.className = 'card chat-step';
    var title = document.createElement('div');
    var b = document.createElement('b');
    b.textContent = '#' + d.index + ' ' + d.action;
    title.appendChild(b);
    var t = document.createElement('span');
    t.textContent = ' on ' + (d.target || 'no target');
    title.appendChild(t);
    card.appendChild(title);
    if (d.reasoning) {
      var r = document.createElement('div');
      r.className = 'muted';
      r.textContent = d.reasoning;
      card.appendChild(r);
    }
    if (d.shot) {
      var img = document.createElement('img');
      img.src = d.shot;
      img.alt = 'screenshot after step ' + d.index;
      card.appendChild(img);
    }
    chat.appendChild(card);
    scrollDown(card);
  }

  function addFinal(d, jobId) {
    var card = document.createElement('div');
    card.className = (d.status === 'completed') ? 'ok' : 'warn';
    var b = document.createElement('b');
    b.textContent = 'Run finished: ' + d.status;
    card.appendChild(b);
    if (d.error) {
      var e = document.createElement('div');
      e.textContent = d.error;
      card.appendChild(e);
    }
    if (d.status === 'completed') {
      card.appendChild(document.createElement('br'));
      var btn = document.createElement('button');
      btn.textContent = 'Save as capability';
      btn.onclick = function () {
        btn.disabled = true;
        fetch('/api/goals/' + jobId + '/save', {method: 'POST'})
          .then(function (r) { return r.json().then(function (j) { return {ok: r.ok, body: j}; }); })
          .then(function (res) {
            if (!res.ok) {
              note('Save failed: ' + (res.body.error || 'unknown error'), 'error');
              btn.disabled = false;
              return;
            }
            var a = document.createElement('a');
            a.href = res.body.url;
            a.textContent = 'Open in catalog: ' + res.body.name;
            card.appendChild(document.createElement('br'));
            card.appendChild(a);
            scrollDown(card);
          });
      };
      card.appendChild(btn);
    }
    chat.appendChild(card);
    scrollDown(card);
    running = false;
  }

  async function refreshKeyWarn() {
    try {
      var r = await fetch('/api/llm-key-status');
      var j = await r.json();
      var show = document.getElementById('client').value === 'openai' && !j.openai_available;
      document.getElementById('keywarn').style.display = show ? 'block' : 'none';
    } catch (err) { /* non-fatal */ }
  }
  document.getElementById('client').addEventListener('change', refreshKeyWarn);
  refreshKeyWarn();

  document.getElementById('run').onclick = async function () {
    if (running) { return; }
    var goal = document.getElementById('goal').value.trim();
    if (!goal) {
      note('Type a goal first.', 'warn');
      return;
    }
    running = true;
    note('You: ' + goal);
    var status = note('Starting the agent run...', 'card muted');
    var res, body;
    try {
      res = await fetch('/api/goals', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          goal: goal,
          client: document.getElementById('client').value,
          headless: document.getElementById('headless').checked
        })
      });
      body = await res.json();
    } catch (err) {
      status.textContent = 'Could not reach the Studio server.';
      running = false;
      return;
    }
    if (!res.ok) {
      status.className = 'error';
      status.textContent = 'Could not start: ' + (body.error || 'unknown error');
      running = false;
      return;
    }
    status.textContent = 'Agent is working. Steps appear below as they happen.';
    var es = new EventSource('/api/goals/' + body.job_id + '/events');
    es.addEventListener('step', function (ev) { addStep(JSON.parse(ev.data)); });
    es.addEventListener('final', function (ev) {
      addFinal(JSON.parse(ev.data), body.job_id);
      es.close();
    });
    es.onerror = function () {
      es.close();
      if (running) {
        note('Lost the event stream. The run may still be going; check the evidence directory.', 'warn');
        running = false;
      }
    };
  };
})();
</script>
"""


def _capability_view(
    catalog: Catalog, path: Path, artifact: CapabilityArtifact
) -> dict[str, Any]:
    """Shared view-model for list, detail, and history pages."""
    record = catalog.registry.get(artifact.id, artifact.version)
    state = catalog._approval_state(artifact)
    history = []
    if record:
        for r in sorted(record.replays, key=lambda x: x.timestamp, reverse=True)[:10]:
            history.append(
                {
                    "timestamp": r.timestamp,
                    "badge": _status_badge(r.result_class),
                    "inputs_hash": r.inputs_hash,
                }
            )
    return {
        "name": artifact.name,
        "url_name": _quote(artifact.name),
        "description": artifact.description,
        "version": artifact.version,
        "approval_badge": _approval_badge(state),
        "approval_state": state,
        "reliability": _reliability_text(record),
        "outputs": [o.name for o in artifact.outputs],
        "history": history,
        "path": str(path),
    }


def create_studio_app(
    capabilities_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    evidence_base: str = "./evidence",
) -> FastAPI:
    """Build the Macro Studio FastAPI app."""
    from bankgpt_cua.evidence import RunLogger, new_run_dir
    from bankgpt_cua.escalation import EscalationManager
    from bankgpt_cua.guardrails import Policy, redact_dict
    from bankgpt_cua.replay import _validate_inputs, replay
    from bankgpt_cua.surface import WebSurface

    registry = Registry(path=registry_path) if registry_path else Registry()
    catalog = Catalog(
        capabilities_dir=capabilities_dir,
        registry=registry,
        evidence_base=evidence_base,
    )
    app = FastAPI(title="Macro Studio")

    def _resolve(name: str) -> tuple[Path, CapabilityArtifact] | HTMLResponse:
        try:
            return catalog.get(name)
        except CatalogError as exc:
            return HTMLResponse(
                _page(
                    "Not found",
                    f"<div class='error'><b>Unknown capability:</b> {exc}</div>"
                    "<p><a href='/'>&larr; all capabilities</a></p>",
                ),
                status_code=404,
            )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        caps = [
            _capability_view(catalog, p, a) for p, a in catalog._scan()
        ]
        body = _LIST_TEMPLATE.render(
            capabilities=caps,
            capabilities_dir=str(catalog.capabilities_dir),
        )
        return _page("Macro Studio", body)

    @app.get("/history", response_class=HTMLResponse)
    def history() -> str:
        caps = [
            _capability_view(catalog, p, a) for p, a in catalog._scan()
        ]
        return _page("Replay history", _HISTORY_TEMPLATE.render(capabilities=caps))

    def _detail_body(
        path: Path, artifact: CapabilityArtifact, unreachable_banner: bool
    ) -> str:
        """Render the full detail page body for a capability."""
        redacted_names = {p.name for p in artifact.inputs if p.redact_in_logs}
        steps = []
        for s in sorted(artifact.steps, key=lambda x: x.index):
            cp = s.checkpoint
            steps.append(
                {
                    "html": _describe_step(s, redacted_names),
                    "rationale": s.target.rationale if s.target else "",
                    "checkpoint": cp is not None,
                    "checkpoint_type": cp.type if cp else "",
                    "checkpoint_value": cp.value if cp else "",
                    "checkpoint_desc": cp.description if cp else "",
                    "notes": s.notes,
                }
            )
        view = _capability_view(catalog, path, artifact)
        entry_point = (artifact.surface or {}).get("entry_point", "")
        return _DETAIL_TEMPLATE.render(
            artifact=artifact,
            url_name=_quote(artifact.name),
            approval_badge=_approval_badge(view["approval_state"]),
            approval_state=view["approval_state"],
            reliability=view["reliability"],
            inputs=[
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "pattern": p.pattern,
                    "description": p.description,
                    "example": p.example,
                    "redacted": p.redact_in_logs,
                }
                for p in artifact.inputs
            ],
            outputs=[
                {
                    "name": o.name,
                    "type": o.type,
                    "description": o.description,
                    "step_index": (o.extraction or {}).get("step_index"),
                    "postprocess": (o.extraction or {}).get("postprocess", "none"),
                }
                for o in artifact.outputs
            ],
            steps=steps,
            success=artifact.success_condition,
            error_policy=[
                {
                    "name": e.name,
                    "outcome_class": e.outcome_class,
                    "match": str(e.match),
                    "code": (e.outcome or {}).get("code", ""),
                    "message": (e.outcome or {}).get("message_template", ""),
                    "recovery": str(e.recovery) if e.recovery else "",
                }
                for e in artifact.error_policy
            ],
            provenance=artifact.provenance,
            history=view["history"],
            unreachable_banner=unreachable_banner,
            entry_point=entry_point,
        )

    @app.get("/capabilities/{name}", response_class=HTMLResponse)
    def detail(name: str, request: Request):
        resolved = _resolve(name)
        if isinstance(resolved, HTMLResponse):
            return resolved
        path, artifact = resolved
        return _page(artifact.name, _detail_body(path, artifact, False))

    @app.post("/capabilities/{name}/replay", response_class=HTMLResponse)
    async def run_replay(name: str, request: Request):
        resolved = _resolve(name)
        if isinstance(resolved, HTMLResponse):
            return resolved
        path, artifact = resolved
        form = await request.form()

        inputs: dict[str, Any] = {}
        for p in artifact.inputs:
            raw = (form.get(f"input:{p.name}") or "").strip()
            inputs[p.name] = raw if raw else None
        headless = form.get("headless") == "on"
        unattended = form.get("unattended") == "on"

        # Validation first: no browser work on bad inputs.
        _, violations = _validate_inputs(artifact, inputs)
        if violations:
            items = "".join(f"<li>{v}</li>" for v in violations)
            return HTMLResponse(
                _page(
                    "Invalid inputs",
                    f"<div class='error'><b>Invalid inputs.</b> Nothing was "
                    f"executed.<ul>{items}</ul></div>"
                    f"<p><a href='/capabilities/{_quote(artifact.name)}'>"
                    "&larr; back</a></p>",
                ),
                status_code=400,
            )

        # Approval gate, same rule as the CLI.
        warning = ""
        if unattended:
            try:
                registry.require_approved(artifact.id, artifact.version)
            except ApprovalError as exc:
                return HTMLResponse(
                    _page(
                        "Approval required",
                        f"<div class='error'><b>Unattended replay refused.</b>"
                        f"<pre>{exc}</pre></div>"
                        f"<p><a href='/capabilities/{_quote(artifact.name)}'>"
                        "&larr; back</a></p>",
                    ),
                    status_code=403,
                )
        else:
            record = registry.get(artifact.id, artifact.version)
            state = record.approval_state if record else artifact.approval_state
            if state != "approved":
                warning = (
                    f"This capability is not approved (state: {state}). "
                    "Running interactively anyway; unattended runs require approval."
                )

        # Mock bank reachability, checked before launching a browser.
        entry_point = (artifact.surface or {}).get("entry_point", "")
        if entry_point and not _check_entry_point(entry_point):
            return _page(artifact.name, _detail_body(path, artifact, True))

        os.makedirs(evidence_base, exist_ok=True)
        run_dir = new_run_dir(evidence_base, "studio-replay")
        secret_values = [
            str(inputs[p.name])
            for p in artifact.inputs
            if p.redact_in_logs and p.name in inputs and inputs[p.name] is not None
        ]

        def _redact(payload: Any) -> Any:
            return redact_dict(payload, extra_values=secret_values)

        def _execute() -> Any:
            # Runs in a worker thread: the replay engine uses Playwright's
            # sync API, which refuses to run inside an asyncio event loop.
            logger = RunLogger(run_dir, redact_fn=_redact)
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
            return result

        from starlette.concurrency import run_in_threadpool

        result = await run_in_threadpool(_execute)

        status = result.status
        if status in ("success", "business_outcome", "recovered", "hard_failure"):
            try:
                registry.record_replay(artifact.id, artifact.version, inputs, status)
            except Exception:
                pass  # the registry must never break a run

        shown_inputs = _masked_inputs(artifact, {k: v for k, v in inputs.items() if v is not None})
        body = _RESULT_TEMPLATE.render(
            url_name=_quote(artifact.name),
            name=artifact.name,
            status_badge=_status_badge(status),
            warning=warning,
            inputs=str(shown_inputs),
            outputs=dict(result.outputs or {}),
            business_outcome=result.business_outcome,
            recoveries=list(result.recoveries_applied or []),
            failure=result.failure,
            run_dir=run_dir,
        )
        return _page(f"Replay result: {artifact.name}", body)

    @app.post("/capabilities/{name}/approve")
    async def approve(name: str, request: Request) -> RedirectResponse:
        resolved = _resolve(name)
        if isinstance(resolved, HTMLResponse):
            return RedirectResponse("/", status_code=303)
        path, artifact = resolved
        form = await request.form()
        reviewer = (form.get("reviewer") or "").strip() or "studio-operator"
        notes = (form.get("notes") or "").strip()
        registry.approve(
            artifact.id, artifact.version, by=reviewer, notes=notes,
            artifact_path=str(path),
        )
        # Mirror onto the artifact file so it stays self-describing.
        artifact.approval_state = "approved"
        artifact.approved_by = reviewer
        from datetime import datetime, timezone

        artifact.approved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        artifact.save(path)
        return RedirectResponse(
            f"/capabilities/{_quote(artifact.name)}", status_code=303
        )

    # ---- Goal chat: run the agent live and stream progress ----

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page() -> str:
        return _page("Goal chat", _CHAT_PAGE)

    @app.get("/api/llm-key-status")
    def api_llm_key_status() -> dict[str, bool]:
        return {"openai_available": goals.llm_key_available()}

    @app.post("/api/goals")
    async def api_start_goal(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Request body must be JSON."}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)
        entry = goals.DEFAULT_CHAT_ENTRY
        if not _check_entry_point(entry):
            return JSONResponse(
                {
                    "error": (
                        f"Mock bank is not reachable at {entry}. "
                        "Start it first: python tools/serve_mock.py"
                    )
                },
                status_code=409,
            )
        try:
            job = goals.start_goal(
                body.get("goal") or "",
                body.get("client") or "mock",
                bool(body.get("headless", True)),
                entry,
                evidence_base=evidence_base,
            )
        except goals.GoalError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except goals.BusyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return {"job_id": job.id}

    @app.get("/api/goals/{job_id}")
    def api_goal_status(job_id: str):
        job = goals.get_job(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown job id."}, status_code=404)
        return {
            "job_id": job.id,
            "goal": job.goal,
            "client": job.client,
            "status": job.status,
            "run_dir": job.run_dir,
            "artifact_id": job.artifact_id,
            "error": job.error,
        }

    @app.get("/api/goals/{job_id}/events")
    def api_goal_events(job_id: str):
        job = goals.get_job(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown job id."}, status_code=404)

        def gen() -> Any:
            while True:
                try:
                    item = job.events.get(timeout=25)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], default=str)}\n\n"
                )
                if item["event"] == "final":
                    break

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/goals/{job_id}/shots/{n}")
    def api_goal_shot(job_id: str, n: int):
        job = goals.get_job(job_id)
        if job is None or not job.run_dir:
            return JSONResponse({"error": "Unknown job."}, status_code=404)
        path = os.path.join(job.run_dir, "screenshots", f"step-{n}.png")
        if not os.path.isfile(path):
            return JSONResponse({"error": "Screenshot not found."}, status_code=404)
        return FileResponse(path, media_type="image/png")

    @app.post("/api/goals/{job_id}/save")
    def api_goal_save(job_id: str):
        job = goals.get_job(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown job id."}, status_code=404)
        try:
            result = goals.save_as_capability(job, str(catalog.capabilities_dir))
        except goals.GoalError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return result

    return app
