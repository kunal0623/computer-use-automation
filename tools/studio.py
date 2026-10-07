"""Serve the Macro Studio web UI.

Usage:
    python tools/studio.py [--port 8771] [--capabilities-dir ./capabilities]

The Studio renders saved capability artifacts for humans: what each
capability does, the steps it takes, how it handles errors, and a form to
replay it. Start the mock bank first (python tools/serve_mock.py) if you
want to run replays from the UI.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Macro Studio web UI.")
    parser.add_argument("--port", type=int, default=8771,
                        help="Port to listen on (default: 8771).")
    parser.add_argument("--capabilities-dir", default=None,
                        help="Capabilities directory (default: ./capabilities).")
    parser.add_argument("--registry", default=None,
                        help="Registry path (default: ./registry.json).")
    parser.add_argument("--evidence-base", default="./evidence",
                        help="Base directory for replay evidence.")
    args = parser.parse_args()

    from bankgpt_cua.studio import create_studio_app

    app = create_studio_app(
        capabilities_dir=args.capabilities_dir,
        registry_path=args.registry,
        evidence_base=args.evidence_base,
    )

    import uvicorn

    print(f"Macro Studio at http://127.0.0.1:{args.port}")
    print("Start the mock bank first for replays: python tools/serve_mock.py")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
