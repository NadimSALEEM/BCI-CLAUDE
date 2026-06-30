"""Background worker for long-running analyses (keeps the UI responsive).

A :class:`AnalysisWorker` runs a callable on a Qt thread and reports progress.
The callable receives ``(progress_cb, is_cancelled)`` so it can publish
percentages and cooperatively stop (e.g. between models during a comparison).
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore

logger = logging.getLogger(__name__)


class AnalysisWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:                       # noqa: D401 (Qt thread entry)
        try:
            result = self._fn(self._emit, self.is_cancelled)
        except Exception as exc:  # noqa: BLE001 - report to the UI, never crash
            logger.exception("Background analysis failed.")
            self.failed.emit(str(exc))
            return
        if not self._cancel:
            self.finished_ok.emit(result)

    def _emit(self, pct: float, msg: str = "") -> None:
        self.progress.emit(int(pct), msg)
