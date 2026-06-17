"""Signal-quality dashboard workspace.

A per-channel table of explainable metrics, colour-coded by rating, with
an overall summary. Every rating is justified by a visible reason and a
visible number -- nothing is hidden, and no impedance values are
fabricated (the Enobio LSL stream does not provide them).
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from neurobci.quality.metrics import (
    QualityRating,
    QualityThresholds,
    compute_quality,
)

_COLOR = {
    QualityRating.GOOD: QtGui.QColor("#1f5f2a"),
    QualityRating.FAIR: QtGui.QColor("#5f5a18"),
    QualityRating.POOR: QtGui.QColor("#6a3d10"),
    QualityRating.BAD: QtGui.QColor("#6a1f1f"),
    QualityRating.NODATA: QtGui.QColor("#333333"),
}

_COLUMNS = ["Channel", "Kind", "Rating", "Std (uV)", "P-P (uV)", "Line %", "HF %", "Reason"]


class SignalQualityWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._thresholds = QualityThresholds()
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.summary = QtWidgets.QLabel("No data.")
        self.summary.setStyleSheet("font-weight:600; padding:4px;")
        top.addWidget(self.summary, stretch=1)
        top.addWidget(QtWidgets.QLabel("Source:"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Preprocessed", True)
        self.source_combo.addItem("Raw", False)
        self.source_combo.setToolTip(
            "Compute quality on the shared preprocessing output (what the BCI "
            "sees) or on the raw stream. Note: a notch/CAR stage will mask the "
            "very line-noise/common artefacts these metrics look for, so use "
            "'Raw' to judge electrode contact.")
        top.addWidget(self.source_combo)
        root.addLayout(top)

        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(len(_COLUMNS) - 1, QtWidgets.QHeaderView.Stretch)
        root.addWidget(self.table, stretch=1)

        disclaimer = QtWidgets.QLabel(
            "Quality is inferred from the signal only (amplitude, variance, "
            "line-noise and high-frequency content). No electrode impedance "
            "is shown because the stream does not provide it."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color:#8a929a; font-style:italic; padding:4px;")
        root.addWidget(disclaimer)

    def update_view(self) -> None:
        engine = self._ctl.engine
        info = engine.stream_info
        if info is None or engine.buffer is None:
            self.summary.setText("No data.")
            self.table.setRowCount(0)
            return

        if self.source_combo.currentData():
            data, _ = engine.latest_processed_seconds(2.0)
        else:
            data, _ = engine.latest_seconds(2.0)
        report = compute_quality(data, info, self._thresholds)
        if not report.channels:
            self.summary.setText("Acquiring… (need ≥ a few samples)")
            return

        src = "preprocessed" if self.source_combo.currentData() else "raw"
        self.summary.setText(
            f"Overall: {report.overall_rating.value.upper()}    "
            f"Bad channels: {report.n_bad_channels} / {len(report.channels)}    "
            f"Window: {report.n_samples} samples @ {report.sfreq:.0f} Hz ({src})"
        )

        self.table.setRowCount(len(report.channels))
        for row, ch in enumerate(report.channels):
            values = [
                ch.name,
                ch.kind,
                ch.rating.value,
                f"{ch.std_uv:.1f}",
                f"{ch.ptp_uv:.0f}",
                f"{ch.line_ratio*100:.0f}",
                f"{ch.hf_ratio*100:.0f}",
                "; ".join(ch.reasons),
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if col in (3, 4, 5, 6):
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                if col == 2:
                    item.setBackground(_COLOR.get(ch.rating, _COLOR[QualityRating.NODATA]))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            len(_COLUMNS) - 1, QtWidgets.QHeaderView.Stretch
        )
