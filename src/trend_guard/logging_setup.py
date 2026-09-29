"""JSONL logging: one file per UTC day under logs/<category>/, rotated automatically."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

# standard LogRecord attributes to exclude when flattening `extra={...}` fields into JSON
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class JsonlFileHandler(logging.Handler):
    """Writes one JSON object per line, rotated to a new file each UTC day."""

    def __init__(self, category: str, base_dir: str | Path = "logs"):
        super().__init__()
        self.category = category
        self.base_dir = Path(base_dir) / category
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._day: str | None = None
        self._fh = None

    def _file_for_today(self):
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        if day != self._day:
            if self._fh:
                self._fh.close()
            self._day = day
            self._fh = open(self.base_dir / f"{day}.jsonl", "a", encoding="utf-8")
        return self._fh

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = str(v)
        fh = self._file_for_today()
        fh.write(json.dumps(payload, default=str) + "\n")
        fh.flush()


_DEFAULT_CATEGORIES = ("api", "decisions", "engine")


def setup_logging(base_dir: str | Path = "logs", categories: tuple[str, ...] = _DEFAULT_CATEGORIES) -> None:
    """Attach a JSONL handler to each named logger. Safe to call more than once."""
    for name in categories:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not any(isinstance(h, JsonlFileHandler) for h in logger.handlers):
            logger.addHandler(JsonlFileHandler(name, base_dir))
