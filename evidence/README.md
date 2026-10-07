# Evidence index

Each directory is one run: `run.jsonl` (redacted step log), `result.json`
or `summary.txt`, `artifact.json` for discoveries, and `screenshots/` +
`dom/` on failures. `RUN_TYPE.txt` in each directory states how the run was
produced.

## Genuine LLM-driven run (the heart of the submission)

- `discover-20261007T223253Z/` — discovery driven by gemini-3.5-flash through
  an OpenAI-compatible endpoint. The model completed the goal in 7 steps
  (log in, search member 12345, read savings balance) and the run was
  recorded as `artifact.json` (`member-savings-balance-lookup`, v1.0.0).
- `replay-20261007T223340Z/` — deterministic replay of that artifact,
  `member_id=12345`: **success**, `outputs: {"savings_balance": "4321.09"}`.
- `replay-20261007T223350Z/` — replay of that artifact, `member_id=99999`:
  **business_outcome** `MEMBER_NOT_FOUND` (record not found is an answer,
  not a crash).
- `replay-20261007T223352Z/` — replay of that artifact, `member_id=abc`:
  **business_outcome** `INVALID_INPUT` (caught by input validation before
  any browser work).

## Scripted pipeline runs (deterministic test doubles)

- `discover-20261007T222202Z/` — same goal driven by the scripted mock
  client, used for pipeline testing while API access was pending.
- `replay-20261007T222205Z`, `replay-20261007T222207Z`,
  `replay-20261007T222209Z` — replays of the scripted artifact (success,
  not-found, invalid-input respectively).

All member IDs are synthetic fixtures and are redacted (`REDACTED:param`)
in every log.
