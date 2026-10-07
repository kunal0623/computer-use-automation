"""Discover a capability: run the agent on the mock bank, then build an artifact.

Usage:
    python tools/discover.py --goal "Check savings balance for member 123"

Flow: create a run dir, run the goal with the agent, and on a completed run
build a CapabilityArtifact via ArtifactBuilder and save artifact.json plus
summary.txt in the run dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_INPUTS = [
    {
        "name": "member_id",
        "type": "string",
        "required": True,
        "description": "Five-digit member ID to look up.",
        "pattern": "^[0-9]{5}$",
        "example": "12345",
        "redact_in_logs": True,
    }
]

DEFAULT_OUTPUTS = [
    {
        "name": "savings_balance",
        "type": "string",
        "description": "Current savings account balance as a decimal string, e.g. 4321.09.",
        "extraction": {
            "step_index": 6,
            "strategy": {"type": "text", "value": "Savings balance"},
            "postprocess": "extract_currency",
        },
    }
]

DEFAULT_ERROR_POLICY = [
    {
        "name": "member_not_found",
        "outcome_class": "business_outcome",
        "match": {"text_contains": ["No member found"]},
        "outcome": {
            "code": "MEMBER_NOT_FOUND",
            "message_template": "No member found for ID {member_id}.",
        },
        "recovery": None,
    },
    {
        "name": "validation_error",
        "outcome_class": "business_outcome",
        "match": {"text_contains": ["must be 5 digits"]},
        "outcome": {
            "code": "INVALID_INPUT",
            "message_template": "The member ID was rejected by the application: invalid format.",
        },
        "recovery": None,
    },
    {
        "name": "session_expired",
        "outcome_class": "recoverable",
        "match": {"text_contains": ["Session expired"]},
        "outcome": {"code": "SESSION_EXPIRED", "message_template": "Session expired."},
        "recovery": {"action": "navigate", "params": {"url": "${entry_point}"}},
    },
    {
        "name": "permission_denied",
        "outcome_class": "business_outcome",
        "match": {"text_contains": ["Permission denied"]},
        "outcome": {
            "code": "PERMISSION_DENIED",
            "message_template": "The operator account is not permitted to view this page.",
        },
        "recovery": None,
    },
]


def _parameterize(artifact, inputs: list) -> None:
    """Replace recorded literal values with ${input} placeholders.

    The discovery run uses concrete values (e.g. member ID 12345). For the
    saved capability to be reusable, any step param or checkpoint that
    exactly matches a declared input's example is rewritten to reference
    the input name. This is a deliberate, minimal canonicalization step:
    replay substitutes the caller's inputs before executing.
    """
    for spec in inputs:
        name = spec.get("name")
        example = spec.get("example")
        if not name or example is None:
            continue
        placeholder = "${" + name + "}"
        example_str = str(example)
        for step in artifact.steps:
            params = step.params or {}
            for key, value in list(params.items()):
                if isinstance(value, str) and value == example_str:
                    params[key] = placeholder
                elif isinstance(value, str) and example_str in value:
                    params[key] = value.replace(example_str, placeholder)
        cond = artifact.success_condition
        if cond and isinstance(cond.value, str) and example_str in cond.value:
            cond.value = cond.value.replace(example_str, placeholder)
        if isinstance(artifact.description, str) and example_str in artifact.description:
            artifact.description = artifact.description.replace(example_str, placeholder)


def _load_json_list(path: str | None, what: str) -> list:
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{what} file must contain a JSON list: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover a capability from the mock bank.")
    parser.add_argument("--goal", required=True, help="Natural-language goal for the agent.")
    parser.add_argument("--entry", default="http://127.0.0.1:8765/login")
    parser.add_argument("--client", choices=["mock", "openai"], default="mock")
    parser.add_argument("--script", default=None, help="JSON list script for ScriptedMockClient.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--headed", action="store_false", dest="headless")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=600, help="Run timeout in seconds.")
    parser.add_argument("--out", default="./evidence", help="Base evidence directory.")
    parser.add_argument("--artifact-id", default=None)
    parser.add_argument("--artifact-name", default=None)
    parser.add_argument("--inputs-spec", default=None, help="JSON list of input param dicts.")
    parser.add_argument("--outputs-spec", default=None, help="JSON list of output def dicts.")
    parser.add_argument("--error-policy-spec", default=None, help="JSON list of error policy entries.")
    parser.add_argument("--operator-port", type=int, default=None,
                        help="If given, serve the operator console on this port in a background thread.")
    args = parser.parse_args()

    from bankgpt_cua.agent import run_goal
    from bankgpt_cua.llm import ScriptedMockClient, default_demo_script, OpenAICompatClient
    from bankgpt_cua.surface import WebSurface
    from bankgpt_cua.guardrails import Policy, redact_dict
    from bankgpt_cua.escalation import EscalationManager, create_operator_app
    from bankgpt_cua.evidence import RunLogger, new_run_dir
    from bankgpt_cua.artifact import ArtifactBuilder

    if args.client == "mock":
        print("NOTE: scripted pipeline client, not a genuine LLM run")
        script = _load_json_list(args.script, "script") if args.script else default_demo_script(args.entry)
        llm = ScriptedMockClient(script)
    else:
        llm = OpenAICompatClient()

    os.makedirs(args.out, exist_ok=True)
    run_dir = new_run_dir(args.out, "discover")
    # Load input specs up front so values flagged redact_in_logs are masked
    # in the run log (the scripted demo reuses the example values).
    _inputs_spec = _load_json_list(args.inputs_spec, "inputs") or DEFAULT_INPUTS
    _secret_values = [
        str(spec["example"])
        for spec in _inputs_spec
        if spec.get("redact_in_logs") and spec.get("example") is not None
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

    if args.operator_port:
        import uvicorn

        app = create_operator_app(escalation_mgr, {"goal": args.goal, "run_dir": run_dir})
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=args.operator_port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        print(f"Operator console at http://127.0.0.1:{args.operator_port}")

    run = run_goal(
        args.goal,
        args.entry,
        llm,
        surface,
        policy=policy,
        escalation_mgr=escalation_mgr,
        logger=logger,
        max_steps=args.max_steps,
        timeout_s=args.timeout,
    )

    status = getattr(run, "status", "unknown")
    run_id = getattr(run, "run_id", os.path.basename(run_dir))
    logger.finalize(run_id)

    artifact_path = os.path.join(run_dir, "artifact.json")
    summary_path = os.path.join(run_dir, "summary.txt")
    summary_lines = [f"goal: {args.goal}", f"status: {status}", f"run_dir: {run_dir}"]

    if status == "completed":
        inputs = _inputs_spec
        outputs = _load_json_list(args.outputs_spec, "outputs") or DEFAULT_OUTPUTS
        error_policy = (
            _load_json_list(args.error_policy_spec, "error policy")
            if args.error_policy_spec
            else DEFAULT_ERROR_POLICY
        )
        artifact = ArtifactBuilder.from_run(
            run,
            artifact_id=args.artifact_id or f"artifact-{run_id}",
            name=args.artifact_name or args.goal,
            inputs=inputs,
            outputs=outputs,
            error_policy=error_policy,
            review_notes="Built by tools/discover.py from a completed agent run.",
        )
        _parameterize(artifact, inputs)
        if hasattr(artifact, "model_dump"):
            payload = artifact.model_dump()
        elif hasattr(artifact, "to_dict"):
            payload = artifact.to_dict()
        else:
            payload = dict(artifact)
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        summary_lines.append(f"artifact: {artifact_path}")
    else:
        summary_lines.append("artifact: not built (run did not complete)")

    with open(summary_path, "w", encoding="utf-8") as handle:
        redacted_summary = "\n".join(summary_lines)
        for secret in _secret_values:
            redacted_summary = redacted_summary.replace(secret, "***")
        handle.write(redacted_summary + "\n")

    print("\n".join(summary_lines))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
