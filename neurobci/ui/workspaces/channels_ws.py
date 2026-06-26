"""Channel configuration & validation workspace.

Works in two modes:

* **Offline** (no stream connected) — edit the *configured* montage from the
  start: set the channel count, fill names from a preset, rename channels to
  real 10-20 positions, and set each kind (EEG / EOG / misc). Applying writes
  the montage back into the configuration, so the next acquisition (and the
  simulator, and the topomaps) use it.
* **Online** (stream connected) — relabel/reclassify the live stream's
  channels when its metadata is wrong, then validate against the expected
  montage. Problems are surfaced explicitly (errors red, warnings amber).
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from neurobci.acquisition.default_montages import MONTAGE_PRESETS, build_stream_info
from neurobci.acquisition.validation import Severity, validate_stream
from neurobci.core.electrodes import classify_channel
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, KIND_MISC

_KINDS = [KIND_EEG, KIND_EOG, KIND_MISC]


class ChannelsWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._kind_combos: list[QtWidgets.QComboBox] = []
        self._built_key: tuple | None = None
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        hint = QtWidgets.QLabel(
            "Double-click a Channel cell to rename it to a real 10-20 position "
            "(e.g. Pz, Oz). Use a preset to relabel generic names, then Apply.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7fb0c8; font-style:italic;")
        root.addWidget(hint)

        # --- montage preset + count (count editable only offline) ------- #
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("— choose a montage —", None)
        for name in MONTAGE_PRESETS:
            self.preset_combo.addItem(name, name)
        preset_row.addWidget(self.preset_combo)
        self.fill_btn = QtWidgets.QPushButton("Fill names from preset")
        self.fill_btn.clicked.connect(self._fill_preset)
        preset_row.addWidget(self.fill_btn)
        preset_row.addSpacing(16)
        self.count_label = QtWidgets.QLabel("Channels:")
        preset_row.addWidget(self.count_label)
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 256)
        preset_row.addWidget(self.count_spin)
        self.resize_btn = QtWidgets.QPushButton("Set count")
        self.resize_btn.clicked.connect(self._resize)
        preset_row.addWidget(self.resize_btn)
        preset_row.addStretch(1)
        root.addLayout(preset_row)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Channel", "Kind"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        root.addWidget(self.table, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply names & kinds")
        self.apply_btn.clicked.connect(self._apply)
        self.delete_btn = QtWidgets.QPushButton("Delete selected channel(s)")
        self.delete_btn.setToolTip(
            "Remove the selected rows from the montage. Apply to make the whole "
            "app (topomaps, spectral, epoching, recording) drop them.")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.issues = QtWidgets.QTextEdit()
        self.issues.setReadOnly(True)
        self.issues.setMaximumHeight(150)
        root.addWidget(self.issues)

    # ----- state helpers ------------------------------------------------- #

    def _is_offline(self) -> bool:
        return self._ctl.engine.stream_info is None

    def _config_montage(self) -> tuple[list[str], list[str]]:
        """(names, kinds) from the configured montage."""
        ch = self._ctl.config.channels
        eeg, eog = set(ch.eeg_channels), set(ch.eog_channels)
        names = ch.all_channels
        kinds = [KIND_EEG if n in eeg else KIND_EOG if n in eog else KIND_EEG
                 for n in names]
        return names, kinds

    # ----- table construction -------------------------------------------- #

    def _set_table(self, names: list[str], kinds: list[str]) -> None:
        self._kind_combos = []
        self.table.setRowCount(len(names))
        for row, (name, kind) in enumerate(zip(names, kinds)):
            idx_item = QtWidgets.QTableWidgetItem(str(row))
            idx_item.setFlags(idx_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, idx_item)
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() | QtCore.Qt.ItemIsEditable)
            name_item.setToolTip("Double-click to rename to a real 10-20 label.")
            # Remember this row's index in the *current* stream so a later
            # deletion can be mapped back to native source columns.
            name_item.setData(QtCore.Qt.UserRole, row)
            self.table.setItem(row, 1, name_item)
            combo = QtWidgets.QComboBox()
            combo.addItems(_KINDS)
            combo.setCurrentText(kind)
            self._kind_combos.append(combo)
            self.table.setCellWidget(row, 2, combo)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _read_table(self) -> tuple[list[str], list[str], list[int]]:
        names, kinds, cur_idx = [], [], []
        for r in range(self.table.rowCount()):
            names.append(self.table.item(r, 1).text().strip() or f"Ch{r + 1}")
            kinds.append(self._kind_combos[r].currentText())
            cur_idx.append(int(self.table.item(r, 1).data(QtCore.Qt.UserRole)))
        return names, kinds, cur_idx

    def _delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Delete channels", "Select one or more rows to delete.")
            return
        if self.table.rowCount() - len(rows) < 1:
            QtWidgets.QMessageBox.warning(
                self, "Delete channels", "At least one channel must remain.")
            return
        for r in rows:
            self.table.removeRow(r)
            del self._kind_combos[r]
        # Renumber the visible '#' column (current-stream index is preserved in
        # each name item's UserRole for the eventual Apply).
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setText(str(r))

    # ----- actions ------------------------------------------------------- #

    def _fill_preset(self) -> None:
        names = MONTAGE_PRESETS.get(self.preset_combo.currentData())
        if not names:
            return
        # Offline a preset defines the whole montage (resize to it); online the
        # channel count is fixed by the stream, so fill as many as line up.
        if self._is_offline():
            kinds = [classify_channel(n) for n in names]
            self._set_table(names, kinds)
            self.count_spin.setValue(len(names))
            return
        n = self.table.rowCount()
        if len(names) != n:
            QtWidgets.QMessageBox.information(
                self, "Channel count differs",
                f"Preset has {len(names)} channels but the stream has {n}. "
                "Filling as many as line up; review before applying.")
        for row in range(min(n, len(names))):
            self.table.item(row, 1).setText(names[row])
            self._kind_combos[row].setCurrentText(classify_channel(names[row]))

    def _resize(self) -> None:
        if not self._is_offline():
            QtWidgets.QMessageBox.information(
                self, "Stream connected",
                "Channel count is fixed by the live stream. Disconnect to edit "
                "the configured montage.")
            return
        n = self.count_spin.value()
        names, kinds, _ = self._read_table()
        if n < len(names):
            names, kinds = names[:n], kinds[:n]
        else:
            for i in range(len(names), n):
                names.append(f"Ch{i + 1}")
                kinds.append(KIND_EEG)
        self._set_table(names, kinds)

    def _apply(self) -> None:
        names, kinds, cur_idx = self._read_table()
        if self._is_offline():
            self._write_config(names, kinds)
            info = build_stream_info(
                self._ctl.config.channels,
                self._ctl.config.acquisition.expected_sfreq, "configured")
            self._built_key = ("offline",) + tuple(names)
            self._validate(info, offline=True)
            return

        engine = self._ctl.engine
        info = engine.stream_info
        deleted = len(names) < info.n_channels
        if deleted:
            # Map remaining current-stream rows back to native source columns
            # and rebuild the live stream around just those channels.
            base = engine.keep_indices or list(range(engine.native_stream_info.n_channels))
            new_keep = [base[j] for j in cur_idx]
            try:
                info = engine.apply_channel_selection(new_keep, names, kinds)
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Delete channels", str(exc))
                return
            self._write_config(names, kinds)
            self._built_key = None          # force a rebuild from the new stream
            self._validate(info)
            return

        # Rename / reclassify only: mutate the live stream-info in place so
        # every consumer (topomap, quality, recording…) updates at once.
        info.channel_names[:] = names
        info.channel_kinds[:] = kinds
        self._built_key = ("online",) + tuple(names)
        self._write_config(names, kinds)
        self._validate(info)

    def _write_config(self, names: list[str], kinds: list[str]) -> None:
        eeg, eog = [], []
        for name, kind in zip(names, kinds):
            (eog if kind == KIND_EOG else eeg).append(name)
        self._ctl.config.channels.eeg_channels = eeg
        self._ctl.config.channels.eog_channels = eog

    def _validate(self, info, offline: bool = False) -> None:
        cfg = self._ctl.config
        report = validate_stream(
            info, cfg.channels, cfg.acquisition.expected_sfreq,
            cfg.acquisition.sfreq_tolerance,
        )
        prefix = "Configured montage — " if offline else ""
        if report.is_valid and not report.warnings:
            self.summary.setText(f"✅ {prefix}stream matches the expected montage.")
        elif report.is_valid:
            self.summary.setText(f"⚠ {prefix}valid with warnings (see below).")
        else:
            self.summary.setText(
                f"❌ {prefix}validation errors — not ready for calibration.")

        colour = {
            Severity.ERROR: "#ff6b6b",
            Severity.WARNING: "#e0c040",
            Severity.OK: "#36c24a",
        }
        lines = [
            f'<span style="color:{colour.get(i.severity, "#ccc")}">'
            f"[{i.severity.value.upper()}]</span> {i.message}"
            for i in report.issues
        ]
        self.issues.setHtml("<br>".join(lines))

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        offline = self._is_offline()
        # Count controls only make sense when editing the configured montage.
        for w in (self.count_label, self.count_spin, self.resize_btn):
            w.setVisible(offline)

        if offline:
            names, kinds = self._config_montage()
            key = ("offline",) + tuple(names)
            if key != self._built_key:
                self._set_table(names, kinds)
                self.count_spin.setValue(len(names))
                self._built_key = key
                self.summary.setText(
                    "No stream connected — editing the configured montage "
                    "(used when you Start acquisition).")
                self.issues.clear()
            return

        info = self._ctl.engine.stream_info
        key = ("online",) + tuple(info.channel_names)
        if key != self._built_key:
            self._set_table(list(info.channel_names), list(info.channel_kinds))
            self._built_key = key
            self._validate(info)
