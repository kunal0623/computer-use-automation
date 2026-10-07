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

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

83 tests: mock-app contracts, LLM client and surface seam, artifact schema
validation, error taxonomy, replay determinism, guardrails, redaction, and
the escalation state machine.

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
- `bankgpt_cua/mock_bank/` : the local legacy-style target application.
- `tools/` : `discover.py`, `replay.py`, `serve_mock.py`.
- `evidence/` : saved runs from the demo path above.
- `REPORT.md` : design write-up (seven sections, per the brief).

## Notes

- Never commit secrets. The mock credentials (`operator` / `operator`) and
  member data are synthetic fixtures, not real PII.
- Values of inputs flagged `redact_in_logs` are masked in all run logs.
