# BankGPT Computer-Use Automation

A system that lets an AI agent operate legacy bank back-office applications that
expose no API. An LLM drives the UI once to accomplish a goal; the run is
recorded as a typed, versioned **capability artifact**; production invocations
**replay that artifact deterministically with no LLM in the loop**.

This is the take-home submission for interface.ai (BankGPT). It covers the full
required thread: goal -> LLM-driven discovery -> saved capability -> deterministic
replay with error handling -> human escalation -> evidence.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
```

No other services are needed. The target application is a local mock bank
included in this repo. All demo data is synthetic.

## Demo path

1. Start the mock bank (leave it running):

```bash
.venv/bin/python tools/serve_mock.py --port 8765
```

2. Run discovery: the agent drives the mock bank to satisfy the goal and saves
a capability artifact. The default client is a scripted pipeline client used
for testing; see "Real LLM run" below for the genuine model-driven path.

```bash
.venv/bin/python tools/discover.py \
  --goal "Look up member 12345 and read their current savings balance" \
  --artifact-id member-savings-balance-lookup \
  --artifact-name "Member Savings Balance Lookup"
```

This writes `evidence/discover-<timestamp>/` containing `run.jsonl` (redacted
step log), `artifact.json`, and `summary.txt`.

3. Replay the artifact deterministically (no LLM), happy path:

```bash
.venv/bin/python tools/replay.py \
  --artifact evidence/discover-<timestamp>/artifact.json \
  --inputs member_id=12345
```

Expected: `status: success` with `outputs: {"savings_balance": "4321.09"}`.

4. Replay an error state (record not found is a business outcome, not a crash):

```bash
.venv/bin/python tools/replay.py \
  --artifact evidence/discover-<timestamp>/artifact.json \
  --inputs member_id=99999
```

Expected: `status: business_outcome` with
`business_outcome: {"code": "MEMBER_NOT_FOUND", ...}`.

Each replay writes `evidence/replay-<timestamp>/` with `run.jsonl`,
`result.json`, and screenshots/DOM snapshots on failure.

## Real LLM run

Discovery accepts any OpenAI-compatible chat-completions endpoint:

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export LLM_MODEL="gemini-3.8-flash"
.venv/bin/python tools/discover.py --client openai --goal "..." --headed
```

Fallbacks: `OPENAI_API_KEY` / `OPENAI_BASE_URL` are honored if the `LLM_*`
variants are unset. Constructor arguments override environment in all cases.
No key is ever logged; the client raises a clear error when no key is set.
`evidence/` contains one genuine model-driven run
(`discover-20261007T223253Z`, gemini-3.5-flash) alongside the earlier
scripted pipeline runs; each directory's `RUN_TYPE.txt` and
`evidence/README.md` say which is which.

## Escalation / operator console

Pass `--operator-port 8766` to `tools/discover.py` to serve the operator
console during a run. If the agent gets stuck, it raises an intervention
request and pauses; the operator takes over the live browser window, acts,
and hands control back from the console. See REPORT.md section 5.

## Approval & registry

Capabilities move through an approval lifecycle: `draft` -> `approved`
(-> `deprecated` for retirement). Unattended execution is gated on approval.

```bash
# approve a capability (also mirrors the state onto the artifact file)
.venv/bin/python tools/approve.py approve \
  --artifact capabilities/member-savings-balance-lookup.json \
  --by "reviewer" --notes "verified against the genuine run"

# send back to draft, or inspect state and reliability
.venv/bin/python tools/approve.py reject --artifact <path> --by "reviewer" --notes "..."
.venv/bin/python tools/approve.py status --artifact <path>
```

The registry (`./registry.json`, local machine state, override with
`BANKGPT_REGISTRY`) records approval state plus replay history per
artifact id and version. Reliability is successful replays over total
replays; business outcomes count as successful (they are correct answers),
only hard failures count against the score. Only input hashes are stored,
never raw values.

Replay in unattended mode (the production path) refuses to run unless the
artifact is approved, exiting non-zero with the exact approval command:

```bash
.venv/bin/python tools/replay.py --artifact <path> --inputs member_id=12345 --unattended
```

Interactive replay (default) runs anyway but prints a warning when the
capability is not approved. Every replay, interactive or not, is recorded
in the registry.

## Capability catalog

The agent-facing surface: saved artifacts in `./capabilities/` (override
with `BANKGPT_CAPABILITIES`) are exposed as named, typed capabilities an
agent can discover and invoke. Invocation validates inputs, requires
approval, and runs the deterministic replay engine with zero LLM calls.

```bash
.venv/bin/python tools/catalog.py list
.venv/bin/python tools/catalog.py invoke \
  --name "Member Savings Balance Lookup" \
  --inputs '{"member_id": "12345"}'
```

Each invocation writes `evidence/catalog-invoke-<timestamp>/` with
`manifest.json`, `result.json`, and a note recording that the call went
through the catalog. See `evidence/catalog-invoke-20261007T230412Z/` for a
real one.

## Macro Studio

A small web UI for understanding and demoing the system: it lists saved
capabilities with approval badges and reliability scores, renders each
artifact in human terms (typed inputs, steps with robustness rationales,
error policy, provenance), and offers a replay form plus an approve button
for draft capabilities. It reuses the catalog, the registry, and the
deterministic replay engine directly; no LLM is ever in the loop.

```bash
python tools/serve_mock.py            # terminal 1: the mock bank
.venv/bin/python tools/studio.py      # terminal 2: the studio
# open http://127.0.0.1:8771
```

Pick a capability, fill in its inputs, and run the replay. Unattended runs
are refused for unapproved capabilities, with the same message the CLI
gives. If the mock bank is not running, the page tells you to start it
instead of failing obscurely. This UI is a comprehension aid, not part of
the graded core.

### Goal chat

The Studio also has a chat page at `/chat`: type a natural-language goal,
pick the client (`mock` for the deterministic scripted stand-in, `openai`
for a real LLM run), and watch the agent drive the mock bank live. Each
step streams to the page as it happens, with the action, what it targeted,
the model's reasoning, and a screenshot.

```bash
python tools/serve_mock.py            # terminal 1: the mock bank
.venv/bin/python tools/studio.py      # terminal 2: the studio
# open http://127.0.0.1:8771/chat
```

Choosing `openai` needs `LLM_API_KEY` set (falls back to `OPENAI_API_KEY`);
without a key the page warns you and refuses to start. When a run
completes, the page offers "Save as capability", which copies the run's
artifact into the capabilities directory so it shows up in the catalog.
Only one run at a time is allowed, since the browser session is shared.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

118 tests: mock-app contracts, LLM client and surface seam, artifact schema
validation, error taxonomy, replay determinism, guardrails, redaction, the
escalation state machine, approval gating and reliability scoring, the
capability catalog, the Macro Studio pages and replay gating, and the goal
chat API (validation, concurrency cap, SSE stream, save-as-capability).

## Layout

- `bankgpt_cua/llm.py` : pluggable LLM client (observe->decide contract),
  OpenAI-compatible client, scripted mock client.
- `bankgpt_cua/surface.py` : surface seam: perceive (screenshot, a11y tree,
  DOM) and act (click, fill, navigate, read, ...). Artifacts depend only on
  this seam.
- `bankgpt_cua/agent.py` : LLM-driven observe->decide->act loop with stuck
  detection.
- `bankgpt_cua/artifact.py` : versioned capability schema, builder, review
  summary.
- `bankgpt_cua/replay.py` : deterministic LLM-free replay engine.
- `bankgpt_cua/errors.py` : error/outcome taxonomy.
- `bankgpt_cua/guardrails.py` : allowlist policy and redaction.
- `bankgpt_cua/escalation.py` : intervention requests, control-transfer
  state machine, operator console.
- `bankgpt_cua/evidence.py` : redacted JSONL run logs and failure snapshots.
- `bankgpt_cua/registry.py` : local JSON registry of approval state and
  replay history, with reliability scoring.
- `bankgpt_cua/catalog.py` : agent-facing capability catalog: discover and
  invoke saved capabilities by name through the deterministic replay engine.
- `bankgpt_cua/studio.py` : Macro Studio web UI over the catalog, including
  the goal chat page.
- `bankgpt_cua/discovery.py` : shared discovery helpers (default specs,
  literal canonicalization, artifact build and save), used by both the
  discover CLI and the chat page.
- `bankgpt_cua/goals.py` : background goal-run jobs for the chat page:
  validation, concurrency cap, per-step SSE events with screenshots, and
  save-as-capability.
- `bankgpt_cua/mock_bank/` : the local legacy-style target application.
- `tools/` : `discover.py`, `replay.py`, `serve_mock.py`, `approve.py`,
  `catalog.py`, `studio.py`.
- `evidence/` : saved runs from the demo path above.
- `capabilities/` : saved artifacts published to the catalog.
- `REPORT.md` : design write-up (seven sections, per the brief).

## Notes

- Never commit secrets. The mock credentials (`operator` / `operator`) and
  member data are synthetic fixtures, not real PII.
- Values of inputs flagged `redact_in_logs` are masked in all run logs.
