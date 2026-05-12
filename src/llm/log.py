"""Append-only JSONL logger for LLM calls, split across two files per run.

Two files are written to log_dir/ for every pipeline run:
  {run_id}_summary.jsonl  — one line per call: item_id, model, tokens, cache, latency, stage
  {run_id}_detail.jsonl   — one line per call: item_id, full input_text, full output_text

item_id is the first 16 hex chars of SHA-256(system + user), enabling joins
between the two tables without duplicating large payloads in the summary file.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path


def _item_id(system: str, user: str) -> str:
    """Compute a short deterministic ID for a (system, user) prompt pair.

    Used to correlate rows in summary vs. detail JSONL files without storing
    the full payload twice.  The first 16 hex chars of SHA-256 give negligible
    collision probability for typical run sizes.

    Args:
        system: The system-role prompt text.
        user: The user-role prompt text.

    Returns:
        A 16-character lowercase hex string.
    """
    return hashlib.sha256((system + "\n" + user).encode()).hexdigest()[:16]


def _write(path: Path, entry: dict) -> None:
    """Append a JSON entry as a single line to path, creating parent dirs if needed.

    Silently swallows OSError so that a logging failure never interrupts the
    pipeline.  Each call opens and closes the file to avoid keeping handles
    open across the full pipeline lifetime.

    Args:
        path: Destination JSONL file path.
        entry: Dict to serialize as a single JSON line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


class LLMLogger:
    """Append-only logger that writes two JSONL files per pipeline run.

    Instantiated once per ``run_pipeline`` call.  Every LLM backend is expected
    to call ``record`` after each API response so that latency, token counts,
    and full payloads are preserved for debugging and cost tracking.
    """

    def __init__(self, log_dir: Path) -> None:
        """Initialize the logger and create unique filenames for this run.

        The run_id embeds a UTC timestamp and a short random suffix so that
        parallel pipeline runs never collide on the same log files.

        Args:
            log_dir: Directory where the two JSONL files will be written.
        """
        self._run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._summary = log_dir / f"{self._run_id}_summary.jsonl"
        self._detail = log_dir / f"{self._run_id}_detail.jsonl"
        self._start: float = 0.0

    def start_call(self) -> float:
        """Record the wall-clock start time for the next LLM call.

        Call this immediately before sending the API request.  The stored
        timestamp is used by ``record`` to compute latency when no explicit
        elapsed value is provided.

        Returns:
            The monotonic timestamp at the moment of the call.
        """
        self._start = time.monotonic()
        return self._start

    def record(
        self,
        *,
        system: str,
        user: str,
        output: str,
        tokens_in: int | None,
        tokens_out: int | None,
        cached_tokens: int | None,
        model_id: str,
        stage: str,
        status: str = "ok",
        error: str | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Write one summary line and one detail line to the JSONL log files.

        Computes the item_id from the prompt content so the two files can be
        joined later.  Never raises: any I/O error is silently ignored so that
        a logging failure cannot abort the pipeline.

        Args:
            system: System-role prompt text sent to the model.
            user: User-role prompt text sent to the model.
            output: The model's decoded response text.
            tokens_in: Input token count, or None if unavailable.
            tokens_out: Output token count, or None if unavailable.
            cached_tokens: Cached-hit token count, or None if unsupported.
            model_id: Identifier of the model that responded.
            stage: Pipeline stage name (e.g. "generate", "lint") for filtering.
            status: ``"ok"`` on success, ``"error"`` otherwise.
            error: Error message string when status is ``"error"``.
            elapsed: Explicit latency in seconds; computed from start_call() if None.
        """
        iid = _item_id(system, user)
        ts = datetime.now(UTC).isoformat()
        lat = round(elapsed or (time.monotonic() - self._start), 3)

        _write(
            self._summary,
            {
                "ts": ts,
                "run_id": self._run_id,
                "item_id": iid,
                "stage": stage,
                "status": status,
                "model_id": model_id,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cached_tokens": cached_tokens,
                "latency_s": lat,
                "error": error,
            },
        )
        _write(
            self._detail,
            {
                "ts": ts,
                "run_id": self._run_id,
                "item_id": iid,
                "stage": stage,
                "status": status,
                "input_text": system + "\n---\n" + user,
                "output_text": output,
                "error": error,
            },
        )
