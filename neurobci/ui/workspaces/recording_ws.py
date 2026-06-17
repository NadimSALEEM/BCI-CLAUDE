"""Recording workspace.

Captures a pseudonymous participant id and free-text notes, starts/stops
writing the live stream to a session directory, shows live recording
statistics, and lets the operator drop event markers (useful for manual
annotation and for exercising the marker pipeline before paradigms exist).
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class RecordingWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        meta = QtWidgets.QGroupBox("Session")
        form = QtWidgets.QFormLayout(meta)
        self.participant = QtWidgets.QLineEdit(self._ctl.config.recording.participant_id)
        self.participant.setPlaceholderText("pseudonymous id (no real names)")
        self.notes = QtWidgets.QLineEdit(self._ctl.config.recording.notes)
        self.dir_label = QtWidgets.QLabel(self._ctl.config.recording.directory)
        form.addRow("Participant id:", self.participant)
        form.addRow("Notes:", self.notes)
        form.addRow("Output dir:", self.dir_label)
        root.addWidget(meta)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("● Start recording")
        self.stop_btn = QtWidgets.QPushButton("■ Stop recording")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.stats = QtWidgets.QLabel("Not recording.")
        self.stats.setStyleSheet("padding:4px;")
        root.addWidget(self.stats)

        # --- markers ----------------------------------------------------- #
        mark_box = QtWidgets.QGroupBox("Markers")
        mark_layout = QtWidgets.QVBoxLayout(mark_box)
        entry = QtWidgets.QHBoxLayout()
        self.marker_edit = QtWidgets.QLineEdit()
        self.marker_edit.setPlaceholderText("marker label")
        self.marker_edit.returnPressed.connect(self._on_marker)
        self.marker_btn = QtWidgets.QPushButton("Mark")
        self.marker_btn.clicked.connect(self._on_marker)
        entry.addWidget(self.marker_edit, stretch=1)
        entry.addWidget(self.marker_btn)
        mark_layout.addLayout(entry)
        self.marker_log = QtWidgets.QListWidget()
        mark_layout.addWidget(self.marker_log)
        root.addWidget(mark_box, stretch=1)

        self._set_recording_ui(False)

    # ----- actions ------------------------------------------------------- #

    def _on_start(self) -> None:
        self._ctl.start_recording(self.participant.text().strip(), self.notes.text())

    def _on_stop(self) -> None:
        self._ctl.stop_recording()

    def _on_marker(self) -> None:
        label = self.marker_edit.text().strip()
        if not label:
            return
        if not self._ctl.engine.is_recording:
            self.marker_log.addItem("(ignored — not recording) " + label)
            return
        self._ctl.engine.push_marker(label)
        n = self._ctl.engine.recorder.stats().n_samples if self._ctl.engine.recorder else 0
        self.marker_log.addItem(f"@sample {n}: {label}")
        self.marker_edit.clear()

    def _set_recording_ui(self, recording: bool) -> None:
        self.start_btn.setEnabled(not recording and self._ctl.engine.running)
        self.stop_btn.setEnabled(recording)
        self.participant.setEnabled(not recording)
        self.notes.setEnabled(not recording)

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        engine = self._ctl.engine
        recording = engine.is_recording
        self._set_recording_ui(recording)
        if recording and engine.recorder is not None:
            st = engine.recorder.stats()
            mb = st.bytes_written / (1024 * 1024)
            self.stats.setText(
                f"● REC  {st.duration_s:6.1f} s   {st.n_samples:,} samples   "
                f"{st.n_markers} markers   {mb:.2f} MB\n{engine.recorder.path}"
            )
        elif not engine.running:
            self.stats.setText("Start acquisition first, then record.")
        else:
            self.stats.setText("Ready to record.")
