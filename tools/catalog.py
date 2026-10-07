"""Agent-facing capability catalog CLI.

Usage:
    python tools/catalog.py list
    python tools/catalog.py invoke --name "<capability name>" --inputs '{"member_id": "12345"}'

The catalog scans ./capabilities/ (override with BANKGPT_CAPABILITIES) for
saved artifact files. Invocation runs the deterministic replay engine with
zero LLM calls and requires the capability to be approved.

Exit codes: 0 on success / business_outcome / recovered, 2 on hard_failure,
1 on catalog errors (unknown capability, not approved, invalid inputs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capability catalog.")
    parser.add_argument("--dir", default=None,
                        help="Capabilities directory (default: ./capabilities).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List capabilities with approval and reliability.")

    p_invoke = sub.add_parser("invoke", help="Invoke a capability by name.")
    p_invoke.add_argument("--name", required=True, help="Capability name or id.")
    p_invoke.add_argument("--inputs", default="{}",
                          help="Inputs as a JSON object string.")
    p_invoke.add_argument("--headless", action="store_true", default=True)
    p_invoke.add_argument("--headed", action="store_false", dest="headless")

    args = parser.parse_args()

    from bankgpt_cua.catalog import Catalog, CatalogError

    catalog = Catalog(capabilities_dir=args.dir)

    if args.command == "list":
        print(json.dumps(catalog.list_capabilities(), indent=2))
        return 0

    try:
        inputs = json.loads(args.inputs)
    except json.JSONDecodeError as exc:
        print(f"error: --inputs must be a JSON object: {exc}", file=sys.stderr)
        return 1
    if not isinstance(inputs, dict):
        print("error: --inputs must be a JSON object", file=sys.stderr)
        return 1

    try:
        result = catalog.invoke(args.name, inputs, headless=args.headless)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    if result.status in ("success", "business_outcome", "recovered"):
        return 0
    if result.status == "hard_failure":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
