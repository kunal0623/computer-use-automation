"""Serve the BankGPT mock bank app over HTTP.

Usage:
    python tools/serve_mock.py [--port 8765]

The port can also be set with the MOCK_PORT environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the mock bank app.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MOCK_PORT", "8765")),
        help="Port to listen on (env MOCK_PORT overrides the default 8765).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    args = parser.parse_args()

    try:
        from bankgpt_cua.mock_bank.app import create_app
    except ImportError:
        try:
            from bankgpt_cua.mock_bank import create_app  # type: ignore[no-redef]
        except ImportError:
            print(
                "error: bankgpt_cua.mock_bank.app.create_app() is not available yet "
                "(the mock app is built by a different worker).",
                file=sys.stderr,
            )
            return 2

    import uvicorn

    app = create_app()
    print(f"Serving mock bank on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
