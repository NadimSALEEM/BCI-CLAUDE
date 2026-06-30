"""Project analysis history (reproducible record of every run).

A small JSON-backed store of :class:`HistoryRecord`s -- each pairs an
:class:`~neurobci.analysis.shared.config.AnalysisConfig` with a results summary
and the paths of any exported artifacts. The Analysis-History panel renders
these and offers re-run / duplicate / export. Persistence is best-effort and
never raises into the UI.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from neurobci.analysis.shared.config import AnalysisConfig

logger = logging.getLogger(__name__)


def default_history_path() -> Path:
    return Path.home() / ".neurobci" / "analysis_history.json"


@dataclass
class HistoryRecord:
    id: str
    config: AnalysisConfig
    summary: dict = field(default_factory=dict)     # short results for the table
    artifacts: dict = field(default_factory=dict)   # {"csv": path, "html": path, ...}

    def to_dict(self) -> dict:
        return {"id": self.id, "config": self.config.to_dict(),
                "summary": self.summary, "artifacts": self.artifacts}

    @classmethod
    def from_dict(cls, d: dict) -> "HistoryRecord":
        return cls(id=d["id"], config=AnalysisConfig.from_dict(d.get("config", {})),
                   summary=d.get("summary", {}), artifacts=d.get("artifacts", {}))

    # convenience accessors used by the table view
    @property
    def created(self) -> str:
        return self.config.created

    @property
    def kind(self) -> str:
        return self.config.kind


class AnalysisHistory:
    """In-memory list of records with optional JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_history_path()
        self._records: list[HistoryRecord] = []
        self.load()

    # ----- mutation ------------------------------------------------------ #

    def add(self, config: AnalysisConfig, summary: dict | None = None,
            artifacts: dict | None = None) -> HistoryRecord:
        rec = HistoryRecord(id=uuid.uuid4().hex[:8], config=config,
                            summary=summary or {}, artifacts=artifacts or {})
        self._records.insert(0, rec)          # newest first
        self.save()
        return rec

    def remove(self, record_id: str) -> None:
        self._records = [r for r in self._records if r.id != record_id]
        self.save()

    def clear(self) -> None:
        self._records = []
        self.save()

    # ----- access -------------------------------------------------------- #

    def all(self) -> list[HistoryRecord]:
        return list(self._records)

    def get(self, record_id: str) -> HistoryRecord | None:
        return next((r for r in self._records if r.id == record_id), None)

    def __len__(self) -> int:
        return len(self._records)

    # ----- persistence (best-effort) ------------------------------------- #

    def load(self) -> None:
        self._records = []
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = [HistoryRecord.from_dict(d) for d in data]
        except Exception:  # noqa: BLE001 - corrupt history must not block the app
            logger.exception("Could not read analysis history at %s", self.path)
            self._records = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = [r.to_dict() for r in self._records]
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001
            logger.exception("Could not write analysis history at %s", self.path)


def summarize_created(iso: str) -> str:
    """Render an ISO timestamp compactly for the history table."""
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return iso
