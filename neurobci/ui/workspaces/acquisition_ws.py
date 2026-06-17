"""Connection & acquisition workspace.

Lets the user pick a source (simulated / LSL), start and stop the data
flow, and inspect live stream metadata and health. It does not touch the
acquisition thread directly -- it asks the main window, which owns the
engine.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from neurobci.acquisition.discovery import discover_streams
from neurobci.core.app_state import ConnectionStatus


class AcquisitionWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # --- source selection -------------------------------------------- #
        box = QtWidgets.QGroupBox("Source")
        form = QtWidgets.QFormLayout(box)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Simulated (no hardware)", "simulated")
        self.source_combo.addItem("Live LSL (Enobio)", "lsl")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.lsl_name = QtWidgets.QLineEdit()
        self.lsl_name.setPlaceholderText("(blank = auto-discover by type 'EEG')")
        self.lsl_name.setEnabled(False)

        # LSL discovery row.
        scan_row = QtWidgets.QHBoxLayout()
        self.scan_btn = QtWidgets.QPushButton("Scan for LSL streams")
        self.scan_btn.clicked.connect(self._on_scan)
        self.discovered = QtWidgets.QComboBox()
        self.discovered.setEnabled(False)
        self.discovered.currentIndexChanged.connect(self._on_pick_discovered)
        scan_row.addWidget(self.scan_btn)
        scan_row.addWidget(self.discovered, stretch=1)

        form.addRow("Source type:", self.source_combo)
        form.addRow("LSL stream name:", self.lsl_name)
        form.addRow("Discover:", self._wrap(scan_row))
        root.addWidget(box)

        # --- controls ---------------------------------------------------- #
        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # --- live info --------------------------------------------------- #
        info_box = QtWidgets.QGroupBox("Stream")
        info_layout = QtWidgets.QVBoxLayout(info_box)
        self.info_label = QtWidgets.QLabel("Not connected.")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        info_layout.addWidget(self.info_label)
        root.addWidget(info_box)
        root.addStretch(1)

    def _wrap(self, layout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    # ----- callbacks ----------------------------------------------------- #

    def _on_source_changed(self) -> None:
        self.lsl_name.setEnabled(self.source_combo.currentData() == "lsl")

    def _on_scan(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning…")
        QtWidgets.QApplication.processEvents()
        try:
            streams = discover_streams(timeout=1.0)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.scan_btn.setEnabled(True)
            self.scan_btn.setText("Scan for LSL streams")

        self.discovered.blockSignals(True)
        self.discovered.clear()
        if not streams:
            self.discovered.addItem("(no streams found)", None)
            self.discovered.setEnabled(False)
        else:
            for s in streams:
                self.discovered.addItem(s.label, s.name)
            self.discovered.setEnabled(True)
        self.discovered.blockSignals(False)
        if streams:
            self._on_pick_discovered()

    def _on_pick_discovered(self) -> None:
        name = self.discovered.currentData()
        if not name:
            return
        # Selecting a discovered stream switches us to the LSL source.
        idx = self.source_combo.findData("lsl")
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        self.lsl_name.setEnabled(True)
        self.lsl_name.setText(name)

    def _on_start(self) -> None:
        self._ctl.start_acquisition(
            source_type=self.source_combo.currentData(),
            lsl_name=self.lsl_name.text().strip(),
        )

    def _on_stop(self) -> None:
        self._ctl.stop_acquisition()

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.source_combo.setEnabled(not running)
        self.scan_btn.setEnabled(not running)
        self.discovered.setEnabled(not running and self.discovered.count() > 0)
        self.lsl_name.setEnabled(not running and self.source_combo.currentData() == "lsl")

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        snap = self._ctl.engine.state.snapshot()
        info = snap.stream_info
        if info is None:
            self.info_label.setText("Not connected.")
            return
        eeg = ", ".join(info.channel_names[i] for i in info.eeg_indices)
        eog = ", ".join(info.channel_names[i] for i in info.eog_indices) or "(none)"
        warn = ""
        if snap.connection in (ConnectionStatus.STALE, ConnectionStatus.LOST):
            warn = f"\n⚠  Connection {snap.connection.value.upper()}."
        self.info_label.setText(
            f"Name: {info.name}   [{info.source_kind}]\n"
            f"Sampling rate: {info.sfreq:.1f} Hz nominal / "
            f"{snap.measured_sfreq:.1f} Hz measured\n"
            f"Channels: {info.n_channels} "
            f"(EEG {len(info.eeg_indices)}, EOG {len(info.eog_indices)})\n"
            f"EEG: {eeg}\n"
            f"EOG: {eog}\n"
            f"Samples received: {snap.samples_received:,}   "
            f"dropped est.: {snap.dropped_samples}"
            f"{warn}"
        )
