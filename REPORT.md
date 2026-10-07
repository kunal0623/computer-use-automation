# Design Report: Computer-Use Automation System

## 1. Architecture

The system is a single Python package with a strict layering: the LLM
discovers, the artifact remembers, the replay engine executes. Four layers
matter.

**Surface seam** (`surface.py`). The only module that touches a browser. It
exposes `perceive()` (URL, title, accessibility tree, DOM excerpt, screenshot)
and `act()` (navigate, click, fill, press, select, wait, read), plus locator
strategy resolution. Everything above it, including the recorded artifact,
speaks only in terms of this seam: actions plus ordered locator strategies.
This is the deliberate portability boundary. A legacy-web or desktop surface
would implement the same perceive/act contract (a11y tree plus coordinates
for desktop) without changing the artifact format or the replay engine.

**Discovery** (`agent.py`, `llm.py`). An observe->decide->act loop: the surface
is perceived, the LLM returns a typed `ActionDecision` (action, target hints,
value, reasoning), guardrails vet it, the surface executes it, and the step is
recorded with the strategy chain that was actually used. The LLM client is a
protocol; the OpenAI-compatible implementation and a scripted mock implement
it. Stuck detection (repeated no-progress steps, step/timeout budgets, or an
explicit escalate decision) routes to the escalation layer instead of looping
forever.

**Capability artifact** (`artifact.py`). A versioned, JSON-serializable
document: ordered steps with locator strategy chains and robustness
rationales, typed inputs, typed outputs with extraction shapes, checkpoints,
an error-handling policy, and provenance. A builder converts a completed run
into an artifact and canonicalizes recorded literals into `${input}`
placeholders so the artifact is reusable, not a transcript.

**Production execution** (`replay.py`). Given an artifact and inputs, it
validates and coerces inputs, substitutes parameters, and walks the steps
with zero LLM calls: resolve strategies in fixed order, act, classify the
resulting state against the error taxonomy, verify checkpoints, extract
declared outputs. It returns a structured result that distinguishes success,
expected business outcomes, recovered conditions, and hard failures.

Trade-offs: one process, no queues or services. The brief explicitly
discourages scaling infrastructure, and a single-process design keeps the
evidence story (one run directory per run) trivially auditable. Playwright
was chosen over screenshot-coordinate control because the mock surface has a
real DOM; the a11y-tree observation and strategy chains are the concession to
the no-clean-DOM reality, and the seam is shaped so a coordinate fallback can
be added as another strategy type. Pydantic v2 carries all schemas so the
artifact is validated on both write and load.

## 2. Artifact schema

The schema is the product: it is what a calling agent invokes and what a
human reviews. Shape (v1.0):

- Identity: `id`, `name`, `version` (semver), `description`, `schema_version`.
- `surface`: kind (`web`), entry point URL, notes. The artifact names the
  surface kind but contains no browser-specific handles.
- `inputs`: typed parameters (`string`/`integer`/`number`/`boolean`) with
  `required`, `pattern`, `example`, and `redact_in_logs`. The member ID is
  redacted in every log.
- `outputs`: typed values with an `extraction` spec: the step index, a
  locator strategy, and a postprocess step (`none`, `strip_currency`,
  `extract_currency`, `strip_whitespace`).
- `steps`: ordered actions. Each step carries a `LocatorTarget`: an ordered
  chain of strategies (`role`, `label`, `placeholder`, `css`, `xpath`,
  `text`, plus `attr`/`image` reserved) and a one-sentence rationale for why
  the chain is robust. Steps also carry optional per-step checkpoints.
- `success_condition`: the terminal checkpoint (e.g. text present
  `MockBank Member ${member_id}`), with input placeholders resolved at
  replay time.
- `error_policy`: named entries mapping observed states to an outcome class
  (`business_outcome`, `recoverable`, `hard_failure`), a result code and
  message template, and an optional recovery action.
- `provenance`: run id, model, timestamp, goal. `review_notes` for the human.

Why this shape: the brief asks for something an agent can call and a human
can review. Inputs/outputs give it a function-like contract; the strategy
chains plus rationales give a reviewer something to judge (a bare selector
list is not reviewable); the error policy makes the taxonomy concrete per
capability instead of global folklore. `CapabilityArtifact.save/load`
round-trips through validation, and `to_summary()` renders a plain-text
review sheet. The JSON Schema is derivable via `json_schema()`.

## 3. Determinism & error handling

Determinism is achieved by removing every decision from replay. Strategy
order is fixed, waits are explicit, there is no randomness and no
wall-clock-dependent branching; the run id is a hash of artifact id, version,
and inputs, so identical invocations are visibly identical. Targeting uses
the recorded chain with a label->control mapping (a text hint usually
resolves to a `<label>`, and fill needs the `<input>`), and the winning
strategy index is logged per step so drift is observable before it is fatal.

Error handling is the taxonomy in `errors.py`, applied after every step and
mapped to the artifact's own error policy:

- **Business outcomes** are legitimate answers: `MEMBER_NOT_FOUND`,
  `INVALID_INPUT`, `PERMISSION_DENIED`. Replay stops and returns the code
  plus a rendered message. Input validation happens before any browser work.
- **Recoverable conditions** apply a bounded recovery from the policy (wait
  and retry, dismiss a known dialog, navigate back to the entry point on
  session expiry) up to a fixed retry budget, and the result reports
  `recovered` with the recoveries applied.
- **Hard failures** stop immediately with step index, what was expected, what
  was observed, and a screenshot plus DOM snapshot saved to the run
  directory. An unrecoverable failure also raises an intervention request so
  a human can take over.

The mock app injects the realistic states: record not found, validation
error, session timeout via `/trigger/timeout`, a JS confirm dialog on the
risky close-account button, a 403 permission page, and artificial slowness.
Evidence includes a replay that returns `MEMBER_NOT_FOUND` and one that
returns `INVALID_INPUT`.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** The artifact never names Playwright. It names
actions and locator strategies, and the replay engine resolves them through
the surface seam. A legacy web app (framesets, table soup, no test IDs) is
the same seam with weaker strategies: the chain degrades from `role` to
`text` to `image` templates, and the rationale field documents why each
chain was chosen. A desktop app implements `perceive()` via the OS
accessibility tree plus screenshots and `act()` via OS-level input; the
strategy types already include `image` for coordinate fallback. The seam
between "how we perceive/act" and "the recorded flow" is exactly the
`surface.py` module boundary.

**Multi-tenant reuse.** Tenants running the same vendor product share the
artifact; per-tenant differences become overrides, not re-recordings. The
mechanism, designed but not built: the artifact's `surface` block gains a
`variant` key (tenant, app version, branding); a variant overlay file
replaces individual strategy chains or constants (e.g. a renamed button)
while inheriting everything else. Drift detection falls out of the replay
evidence: the winning-strategy index logged per step is a per-tenant
stability signal, and a step that starts resolving on fallback strategies
flags the variant for review. Parameterization (`${member_id}`) already
separates tenant data from flow, which is the prerequisite for all of this.

## 5. Escalation & handoff

Stuck is detected, not guessed: the agent loop watches for repeated
no-progress steps (same action and URL with no state change), exhausted
step/timeout budgets, guardrail denials, and explicit LLM escalate
decisions. On any of these it builds an `InterventionRequest` carrying the
goal, current step, reason, state summary, and screenshot, then moves the
control state machine from AUTO to PAUSED.

The handoff keeps the **same live session**: the browser context the
automation was using stays open (headed mode for a human operator), and a
minimal operator console (a local web app) shows the request, offers
take-over / hand-back controls, and reflects the control state. The human
acts directly in the live window; a small injected recorder captures their
clicks and inputs while the state is HUMAN, and the console exports those
actions as evidence. Hand-back returns the state machine to AUTO and the run
resumes or completes with the human's actions in the log. Illegal
transitions (e.g. hand-back without take-over) are rejected.

What is real: the state machine, the pause/resume on the same session, the
action recorder, and the console endpoints. What is mocked: the console UI
itself is minimal, and there is no real-time co-browsing. A full operator
console is explicitly out of scope in the brief; the seam (request object,
state machine, recorder) is where a real one would attach.

## 6. Safety

The guardrail model has three parts. First, an explicit **allowlist**:
permitted route patterns and action types, evaluated before every action in
both discovery and replay; navigation is checked against the destination
URL, not the page being left. Anything outside is denied with a reason, and
the denial is a terminal, debuggable result. Second, **risky-action
classification**: actions matching configured risky rules (e.g. clicking
"Close savings account", or any action flagged irreversible) are blocked in
`block` mode or parked for human confirmation in `confirm` mode; the demo
uses block mode. Third, **redaction**: regex rules mask SSNs, card numbers,
and bearer tokens, and any input flagged `redact_in_logs` has its runtime
value masked everywhere in logs and summaries. Verified: zero occurrences of
the member ID in the shipped run logs.

Limits, stated honestly: regex redaction is heuristic and cannot catch every
PII shape; risky classification is target-text based and would miss a
redesigned button; the allowlist is per-run configuration, so a misconfigured
policy is a single point of failure. Credentials for the demo are synthetic
fixtures; a production deployment would inject them from a vault at replay
time and never record them. The write-up and code comments mark these limits
where they live.

## 7. Cuts

Deliberately thin or mocked, with reasons:

- **Genuine LLM discovery run.** Done: `evidence/discover-20261007T223253Z/`
  is a real model-driven run (gemini-3.5-flash via an OpenAI-compatible
  endpoint; the scripted client was used only for earlier pipeline runs,
  which are labeled as such). Notes from doing it: the sandbox egress proxy
  required `trust_env=False` with an explicit proxy plus the sandbox CA
  bundle in `llm.py`, and the client retries transient 429/503s with
  backoff. The default model moved from gemini-2.5-flash to gemini-3.8-flash
  (2.5 is retired for new users); the run itself used gemini-3.5-flash
  because 3.8 was overloaded at the time. Model choice stays a config flag.
- **Operator console.** Minimal but functional; a polished co-browsing UI is
  out of scope per the brief.
- **Desktop / legacy surfaces.** Designed at the seam, not implemented; one
  concrete surface was the right depth trade.
- **Multi-tenant overlays and drift dashboards.** Design in section 4;
  building them would be the premature infrastructure the brief warns
  against.
- **Stretch goals** (capability catalog API, code generation, approval
  gating, bounded LLM fallback, cross-tenant canonicalization demos,
  multi-run stability): none taken. The one partial exception is
  canonicalization: discovery rewrites recorded literals into `${input}`
  placeholders, the minimum needed for a reusable artifact.

With more time: the real LLM run first, then approval-gated replay
(draft -> approved) since that is the natural safety complement to the
guardrails, then variant overlays for a second mock tenant to prove the
multi-tenant design.
