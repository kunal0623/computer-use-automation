"""Evidence logging for BankGPT runs.

``RunLogger`` writes a redacted JSONL event stream plus screenshot and DOM
snapshots into one run directory. ``finalize`` writes a manifest.json that
indexes everything, so a run can be audited or replayed later without
trusting the live system.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from bankgpt_cua.guardrails import redact_dict


def new_run_dir(base: str, name: str) -> str:
    """Create ``base/<name>-<utc timestamp>`` and return its path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(base, f"{name}-{stamp}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(base, f"{name}-{stamp}-{counter}")
        counter += 1
    os.makedirs(path, exist_ok=False)
    return path


class RunLogger:
    """Append-only evidence logger for one run."""

    def __init__(
        self,
        run_dir: str,
        redact_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.screenshots_dir = os.path.join(run_dir, "screenshots")
        self.dom_dir = os.path.join(run_dir, "dom")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.dom_dir, exist_ok=True)
        self._jsonl_path = os.path.join(run_dir, "run.jsonl")
        self._redact_fn = redact_fn or (lambda payload: redact_dict(payload))
        self._event_count = 0
        self._files: list[str] = []

    def log(self, event: str, payload: dict) -> dict:
        """Append one redacted event to run.jsonl and return the record."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": self._redact_fn(payload),
        }
        self._event_count += 1
        with open(self._jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record

    def screenshot(self, name: str, png_bytes: bytes) -> str:
        """Save a PNG screenshot under screenshots/ and return its path."""
        filename = name if name.endswith(".png") else f"{name}.png"
        path = os.path.join(self.screenshots_dir, filename)
        with open(path, "wb") as handle:
            handle.write(png_bytes)
        self._files.append(os.path.relpath(path, self.run_dir))
        return path

    def dom(self, name: str, html: str) -> str:
        """Save an HTML DOM snapshot under dom/ and return its path."""
        filename = name if name.endswith(".html") else f"{name}.html"
        path = os.path.join(self.dom_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        self._files.append(os.path.relpath(path, self.run_dir))
        return path

    def finalize(self, run_id: str) -> dict:
        """Write manifest.json indexing the run and return it."""
        manifest = {
            "run_id": run_id,
            "events": self._event_count,
            "files": list(self._files),
        }
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        return manifest
