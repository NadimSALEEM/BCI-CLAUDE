"""Channel configuration & validation workspace.

Shows the live stream's channels, lets the user correct each channel's
kind (EEG / EOG / misc) when the stream metadata is wrong, and validates
the montage/rate against the configured expectation. Validation problems
are surfaced explicitly (errors in red, warnings in amber) rather than
being silently fixed.
"""

from __future__ import annotations

from PyQt5 import QtGui, QtWidgets

from neurobci.acquisition.validation import Severity, validate_stream
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, KIND_MISC

_KINDS = [KIND_EEG, KIND_EOG, KIND_MISC]


class ChannelsWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._kind_combos: list[QtWidgets.QComboBox] = []
        self._built_for: list[str] = []
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        self.summary = QtWidgets.QLabel("Connect a stream to validate channels.")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Channel", "Kind"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply kinds & re-validate")
        self.apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(self.apply_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.issues = QtWidgets.QTextEdit()
        self.issues.setReadOnly(True)
        self.issues.setMaximumHeight(150)
        root.addWidget(self.issues)

    # ----- table construction -------------------------------------------- #

    def _ensure_table(self, info) -> None:
        if info.channel_names == self._built_for:
            return
        self._built_for = list(info.channel_names)
        self._kind_combos = []
        self.table.setRowCount(info.n_channels)
        for row in range(info.n_channels):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row)))
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(info.channel_names[row])
            )
            combo = QtWidgets.QComboBox()
            combo.addItems(_KINDS)
            combo.setCurrentText(info.channel_kinds[row])
            self._kind_combos.append(combo)
            self.table.setCellWidget(row, 2, combo)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    # ----- actions ------------------------------------------------------- #

    def _apply(self) -> None:
        info = self._ctl.engine.stream_info
        if info is None:
            return
        # Mutate the live stream-info kinds in place so quality/recording use
        # the corrected layout immediately.
        new_kinds = [c.currentText() for c in self._kind_combos]
        info.channel_kinds[:] = new_kinds
        # Reflect EOG selection back into the persisted channel config.
        eeg, eog = [], []
        for name, kind in zip(info.channel_names, new_kinds):
            (eog if kind == KIND_EOG else eeg).append(name)
        self._ctl.config.channels.eeg_channels = eeg
        self._ctl.config.channels.eog_channels = eog
        self._validate(info)

    def _validate(self, info) -> None:
        cfg = self._ctl.config
        report = validate_stream(
            info, cfg.channels, cfg.acquisition.expected_sfreq,
            cfg.acquisition.sfreq_tolerance,
        )
        if report.is_valid and not report.warnings:
            self.summary.setText("✅ Stream matches the expected montage.")
        elif report.is_valid:
            self.summary.setText("⚠ Valid with warnings (see below).")
        else:
            self.summary.setText("❌ Validation errors — not ready for calibration.")

        lines = []
        colour = {
            Severity.ERROR: "#ff6b6b",
            Severity.WARNING: "#e0c040",
            Severity.OK: "#36c24a",
        }
        for issue in report.issues:
            c = colour.get(issue.severity, "#cccccc")
            lines.append(
                f'<span style="color:{c}">[{issue.severity.value.upper()}]</span> '
                f"{issue.message}"
            )
        self.issues.setHtml("<br>".join(lines))

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        info = self._ctl.engine.stream_info
        if info is None:
            self.summary.setText("Connect a stream to validate channels.")
            self.table.setRowCount(0)
            self._built_for = []
            self.issues.clear()
            return
        rebuilt = info.channel_names != self._built_for
        self._ensure_table(info)
        if rebuilt:
            self._validate(info)
