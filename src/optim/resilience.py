"""
Resilience helpers: incremental JSONL result writing, resume-from-checkpoint,
full-traceback logging.

A 2-week critical path cannot tolerate 2-hour eval runs that lose everything
on a Windows memory hiccup at sample 847/1000. Every script MUST:

    1. Append each sample result to a .jsonl file immediately.
    2. At startup, read the .jsonl back and skip any already-completed sample_id.
    3. Wrap its main loop in try/except and dump the full traceback to a
       logs/ file with timestamp + model + split, so post-mortems don't
       require scrollback from a long-dead terminal.

This module provides the building blocks; scripts compose them.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


# ---------------------------------------------------------------------------
# JSONL append writer.
# ---------------------------------------------------------------------------

class JsonlAppender:
    """Append-only JSONL writer with flush-on-write semantics.

    Opens the file in append mode, writes one JSON object per line, flushes
    after every write so a crash in the next sample doesn't lose the current
    one. Tolerates being opened on an existing file (resume).
    """

    def __init__(self, path: Path, id_key: str = "sample_id"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.id_key = id_key
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        for r in records:
            self.write(r)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def resume_completed_ids(path: Path, id_key: str = "sample_id") -> Set[str]:
    """Read a JSONL file and return the set of sample_ids already completed.

    Tolerates truncated/corrupted final lines (returns what it can parse).
    Returns an empty set if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return set()
    done: Set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Truncated line from a crash — skip and move on.
                continue
            sid = obj.get(id_key)
            if sid is not None:
                done.add(str(sid))
    return done


# ---------------------------------------------------------------------------
# Traceback logging.
# ---------------------------------------------------------------------------

def configure_traceback_logging(
    log_dir: Path,
    run_name: str,
    also_stdout: bool = True,
) -> logging.Logger:
    """Set up a logger that writes to logs/{run_name}_{timestamp}.log.

    Also installs a sys.excepthook that dumps the full traceback to the log
    file before the process dies. Returns the logger.

    Call this ONCE per script at startup.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{ts}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers so re-configuration is idempotent.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    if also_stdout:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        stream.setLevel(logging.INFO)
        logger.addHandler(stream)

    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        # Still print to stderr so the terminal shows it.
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook
    logger.info("Logging to %s", log_path)
    return logger
