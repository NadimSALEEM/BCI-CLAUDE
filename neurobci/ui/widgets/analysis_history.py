"""Analysis History panel.

Lists every recorded analysis (stats or ML) with its configuration and a short
results summary, and offers per-record inspect / export-config / duplicate /
delete. Backed by :class:`~neurobci.analysis.shared.history.AnalysisHistory`,
so the list survives restarts. Full one-click re-run wiring into the originating
tab is incremental; export-config already makes every run reproducible.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from neurobci.analysis.shared.export import save_json
from neurobci.analysis.shared.history import summarize_created

_COLS = ["When", "Kind", "Type", "Conditions", "Summary"]


class AnalysisHistoryWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._build()
        self.refresh()

    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("<b>Analysis History</b> — every run is "
                                       "saved with its full, reproducible config."))
        bar.addStretch(1)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)
        lay.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)

        btns = QtWidgets.QHBoxLayout()
        for label, slot in (("Inspect config", self._inspect),
                            ("Export config (JSON)", self._export),
                            ("Duplicate", self._duplicate),
                            ("Delete", self._delete),
                            ("Clear all", self._clear)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            btns.addWidget(b)
        lay.addLayout(btns)

        self.detail = QtWidgets.QTextBrowser()
        self.detail.setMaximumHeight(220)
        lay.addWidget(self.detail)

    # ----- data ---------------------------------------------------------- #
    def refresh(self) -> None:
        records = self._ctl.analysis_history.all()
        self.table.setRowCount(len(records))
        for row, rec in enumerate(records):
            cfg = rec.config
            conds = ", ".join(cfg.selection.get("conditions", [])) \
                if isinstance(cfg.selection.get("conditions"), list) else ""
            summary = "; ".join(f"{k}={v}" for k, v in rec.summary.items())
            values = [summarize_created(cfg.created), cfg.kind, cfg.analysis_type,
                      conds, summary]
            for col, val in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setData(QtCore.Qt.UserRole, rec.id)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _selected_id(self):
        items = self.table.selectedItems()
        return items[0].data(QtCore.Qt.UserRole) if items else None

    def _inspect(self) -> None:
        rec = self._ctl.analysis_history.get(self._selected_id())
        if rec:
            self.detail.setText(rec.config.to_json())

    def _export(self) -> None:
        rec = self._ctl.analysis_history.get(self._selected_id())
        if not rec:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export config", f"{rec.config.analysis_type}.json",
            "JSON (*.json)")
        if path:
            save_json(rec.config.to_dict(), path)
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved:\n{path}")

    def _duplicate(self) -> None:
        rec = self._ctl.analysis_history.get(self._selected_id())
        if not rec:
            return
        import dataclasses
        clone = dataclasses.replace(rec.config, notes=(rec.config.notes +
                                                       " [duplicate]").strip())
        self._ctl.analysis_history.add(clone, summary=dict(rec.summary))
        self.refresh()

    def _delete(self) -> None:
        rid = self._selected_id()
        if rid:
            self._ctl.analysis_history.remove(rid)
            self.refresh()

    def _clear(self) -> None:
        if QtWidgets.QMessageBox.question(
                self, "Clear all", "Remove every history record?") \
                == QtWidgets.QMessageBox.Yes:
            self._ctl.analysis_history.clear()
            self.refresh()

    def update_view(self) -> None:
        pass
