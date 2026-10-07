"""Replay a capability artifact against the mock bank.

Usage:
    python tools/replay.py --artifact ./evidence/discover-.../artifact.json --inputs member_id=123

Exit codes: 0 on success / business_outcome / recovered, 2 on hard_failure,
1 on anything else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUCCESS_STATUSES = {"success", "business_outcome", "recovered"}


def _parse_inputs(pairs: list[str]) -> dict:
    inputs: dict = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--inputs entries must be key=value, got: {pair!r}")
        try:
            value: object = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            value = raw
        inputs[key] = value
    return inputs


def _dumpable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[union-attr]
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[union-attr]
    if isinstance(value, dict):
        return {k: _dumpable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dumpable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a capability artifact.")
    parser.add_argument("--artifact", required=True, help="Path to artifact.json.")
    parser.add_argument("--inputs", action="append", default=[],
                        help="Input as key=value; repeatable.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--headed", action="store_false", dest="headless")
    parser.add_argument("--out", default="./evidence", help="Base evidence directory.")
    args = parser.parse_args()

    from bankgpt_cua.artifact import CapabilityArtifact
    from bankgpt_cua.surface import WebSurface
    from bankgpt_cua.guardrails import Policy, redact_dict
    from bankgpt_cua.escalation import EscalationManager
    from bankgpt_cua.evidence import RunLogger, new_run_dir
    from bankgpt_cua.replay import replay

    inputs = _parse_inputs(args.inputs)
    artifact = CapabilityArtifact.load(args.artifact)

    os.makedirs(args.out, exist_ok=True)
    run_dir = new_run_dir(args.out, "replay")
    _secret_values = [
        str(inputs[p.name])
        for p in artifact.inputs
        if p.redact_in_logs and p.name in inputs and inputs[p.name] is not None
    ]

    def _redact(payload):
        return redact_dict(payload, extra_values=_secret_values)

    logger = RunLogger(run_dir, redact_fn=_redact)
    try:
        surface = WebSurface(headless=args.headless)
    except TypeError:
        surface = WebSurface()
    policy = Policy()
    escalation_mgr = EscalationManager(evidence_dir=run_dir)

    result = replay(
        artifact,
        inputs,
        surface,
        policy=policy,
        escalation_mgr=escalation_mgr,
        logger=logger,
        headless=args.headless,
    )

    status = getattr(result, "status", "unknown")
    outputs = _dumpable(getattr(result, "outputs", {}))
    recoveries = _dumpable(getattr(result, "recoveries", []))
    failure = _dumpable(getattr(result, "failure", None))
    business_outcome = _dumpable(getattr(result, "business_outcome", None))

    logger.finalize(getattr(result, "run_id", os.path.basename(run_dir)))
    result_path = os.path.join(run_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"status": status, "outputs": outputs, "recoveries": recoveries,
             "failure": failure, "business_outcome": business_outcome,
             "run_dir": run_dir},
            handle,
            indent=2,
            default=str,
        )

    print(json.dumps(
        {"status": status, "outputs": outputs, "recoveries": recoveries,
         "failure": failure, "business_outcome": business_outcome,
         "result_path": result_path},
        indent=2,
        default=str,
    ))
    if status in SUCCESS_STATUSES:
        return 0
    if status == "hard_failure":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
