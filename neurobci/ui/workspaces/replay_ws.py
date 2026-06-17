"""Replay workspace.

Load a recorded session and play it back through the engine (so every other
tab works on the replayed data), with transport controls (play/pause, speed,
seek, loop), optional artifact injection, and optional re-publication as a
virtual LSL stream.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets

from neurobci.acquisition.artifacts_inject import (
    ArtifactInjectionConfig,
    ArtifactInjector,
)
from neurobci.acquisition.replay_source import ReplaySource
from neurobci.recording.exporter import load_session

logger = logging.getLogger(__name__)


class ReplayWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._session_path: str | None = None
        self._publisher = None
        self._seek_guard = False
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # --- session selection ------------------------------------------ #
        sel = QtWidgets.QHBoxLayout()
        self.browse_btn = QtWidgets.QPushButton("Load session…")
        self.browse_btn.clicked.connect(self._browse)
        self.path_label = QtWidgets.QLabel("No session loaded.")
        self.path_label.setWordWrap(True)
        sel.addWidget(self.browse_btn)
        sel.addWidget(self.path_label, stretch=1)
        root.addLayout(sel)
        self.meta_label = QtWidgets.QLabel("")
        self.meta_label.setWordWrap(True)
        root.addWidget(self.meta_label)

        # --- transport --------------------------------------------------- #
        trans = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start replay")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setEnabled(False)
        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.restart_btn.clicked.connect(self._restart)
        self.restart_btn.setEnabled(False)
        self.speed_combo = QtWidgets.QComboBox()
        for s in ("0.5", "1", "2", "4"):
            self.speed_combo.addItem(f"{s}x", float(s))
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed)
        self.loop_chk = QtWidgets.QCheckBox("Loop")
        for w in (self.start_btn, self.stop_btn, self.pause_btn, self.restart_btn):
            trans.addWidget(w)
        trans.addWidget(QtWidgets.QLabel("Speed:"))
        trans.addWidget(self.speed_combo)
        trans.addWidget(self.loop_chk)
        trans.addStretch(1)
        root.addLayout(trans)

        # --- seek -------------------------------------------------------- #
        seekrow = QtWidgets.QHBoxLayout()
        self.seek = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.sliderReleased.connect(self._on_seek)
        self.pos_label = QtWidgets.QLabel("0.0 / 0.0 s")
        seekrow.addWidget(self.seek, stretch=1)
        seekrow.addWidget(self.pos_label)
        root.addLayout(seekrow)

        # --- artifacts + LSL -------------------------------------------- #
        extra = QtWidgets.QGroupBox("Inject artifacts / publish")
        form = QtWidgets.QFormLayout(extra)
        self.artifact_chk = QtWidgets.QCheckBox("Inject artifacts")
        self.artifact_chk.stateChanged.connect(self._apply_artifacts)
        self.line_uv = QtWidgets.QDoubleSpinBox()
        self.line_uv.setRange(0, 100); self.line_uv.setValue(15.0)
        self.line_uv.setSuffix(" uV line")
        self.line_uv.valueChanged.connect(self._apply_artifacts)
        self.blink_rate = QtWidgets.QDoubleSpinBox()
        self.blink_rate.setRange(0, 5); self.blink_rate.setValue(0.5)
        self.blink_rate.setSuffix(" blinks/s")
        self.blink_rate.valueChanged.connect(self._apply_artifacts)
        self.publish_chk = QtWidgets.QCheckBox("Publish as virtual LSL stream")
        self.publish_chk.stateChanged.connect(self._toggle_publish)
        form.addRow(self.artifact_chk)
        form.addRow(self.line_uv, self.blink_rate)
        form.addRow(self.publish_chk)
        root.addWidget(extra)
        root.addStretch(1)

    # ----- helpers ------------------------------------------------------- #

    def _source(self) -> ReplaySource | None:
        src = self._ctl.engine.source
        return src if isinstance(src, ReplaySource) else None

    def set_session(self, path: str) -> None:
        self._session_path = path
        self.path_label.setText(path)
        try:
            s = load_session(path)
            self.meta_label.setText(
                f"{s.n_samples} samples @ {s.sfreq:.0f} Hz "
                f"({s.n_samples / s.sfreq:.1f} s), {len(s.channel_names)} ch, "
                f"{len(s.markers)} markers — participant {s.meta.get('participant_id')}")
        except Exception as exc:  # noqa: BLE001
            self.meta_label.setText(f"Could not read metadata: {exc}")

    # ----- actions ------------------------------------------------------- #

    def _browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a recorded session directory",
            self._ctl.config.recording.directory)
        if d:
            self.set_session(d)

    def _start(self) -> None:
        if not self._session_path:
            QtWidgets.QMessageBox.information(self, "No session", "Load a session first.")
            return
        acq = self._ctl.config.acquisition
        acq.replay_path = self._session_path
        acq.replay_speed = self.speed_combo.currentData()
        acq.replay_loop = self.loop_chk.isChecked()
        self._ctl.start_acquisition("replay")
        self._apply_artifacts()
        self._set_running(self._ctl.engine.running)

    def _stop(self) -> None:
        self._ctl.stop_acquisition()
        self._set_running(False)

    def _toggle_pause(self) -> None:
        src = self._source()
        if src is None:
            return
        if src.paused:
            src.resume(); self.pause_btn.setText("Pause")
        else:
            src.pause(); self.pause_btn.setText("Resume")

    def _restart(self) -> None:
        if self._source():
            self._source().restart()

    def _on_speed(self) -> None:
        if self._source():
            self._source().set_speed(self.speed_combo.currentData())

    def _on_seek(self) -> None:
        src = self._source()
        if src is not None:
            src.seek(self.seek.value() / 1000.0 * src.duration_s)

    def _apply_artifacts(self) -> None:
        src = self._source()
        if src is None:
            return
        if self.artifact_chk.isChecked():
            cfg = ArtifactInjectionConfig(
                enabled=True, line_uv=self.line_uv.value(),
                blink_rate_hz=self.blink_rate.value())
            src.set_injector(ArtifactInjector(src.info, cfg))
        else:
            src.set_injector(None)

    def _toggle_publish(self) -> None:
        if self.publish_chk.isChecked():
            if not self._session_path:
                self.publish_chk.setChecked(False)
                return
            try:
                from neurobci.acquisition.lsl_publisher import LSLPublisher
                src = ReplaySource(load_session(self._session_path), loop=True)
                self._publisher = LSLPublisher(src, stream_name="NeuroBCI-Replay")
                self._publisher.start()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to publish virtual LSL.")
                QtWidgets.QMessageBox.critical(self, "LSL error", str(exc))
                self.publish_chk.setChecked(False)
        elif self._publisher is not None:
            self._publisher.stop()
            self._publisher = None

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        for w in (self.stop_btn, self.pause_btn, self.restart_btn):
            w.setEnabled(running)

    # ----- refresh ------------------------------------------------------- #

    def update_view(self) -> None:
        src = self._source()
        running = src is not None
        self._set_running(running)
        if src is None:
            return
        self.pos_label.setText(f"{src.position_s:.1f} / {src.duration_s:.1f} s")
        if not self.seek.isSliderDown() and src.duration_s > 0:
            self.seek.blockSignals(True)
            self.seek.setValue(int(1000 * src.position_s / src.duration_s))
            self.seek.blockSignals(False)
