"""Main application window.

Owns the :class:`AcquisitionEngine` and drives all views from a *single*
refresh timer. The timer pulls state/data snapshots and hands them to the
currently-visible workspace; acquisition itself runs on its own thread and
is unaffected by how fast (or slow) the UI repaints.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets

from neurobci.acquisition.engine import AcquisitionEngine
from neurobci.config.schema import AppConfig
from neurobci.quality.metrics import QualityRating, compute_quality
from neurobci.ui.widgets.status_bar import StatusBar
from neurobci.ui.workspaces.acquisition_ws import AcquisitionWorkspace
from neurobci.ui.workspaces.analysis_ws import AnalysisWorkspace
from neurobci.ui.workspaces.calibration_ws import CalibrationWorkspace
from neurobci.ui.workspaces.channels_ws import ChannelsWorkspace
from neurobci.ui.workspaces.control_ws import ControlWorkspace
from neurobci.ui.workspaces.preprocessing_ws import PreprocessingWorkspace
from neurobci.ui.workspaces.raw_eeg_ws import RawEEGWorkspace
from neurobci.ui.workspaces.recording_ws import RecordingWorkspace
from neurobci.ui.workspaces.replay_ws import ReplayWorkspace
from neurobci.ui.workspaces.signal_quality_ws import SignalQualityWorkspace
from neurobci.ui.workspaces.spectral_ws import SpectralWorkspace
from neurobci.version import APP_NAME, __version__

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: AcquisitionEngine, config: AppConfig) -> None:
        super().__init__()
        self.engine = engine
        self.config = config
        self._tick = 0
        self._last_quality: QualityRating | None = None
        self._quality_every = max(1, int(config.ui.refresh_hz / 4))
        self.model = None                     # last calibrated ParadigmModel
        self.calibration_result = None

        self.setWindowTitle(f"{APP_NAME} {__version__} — EEG / BCI platform")
        self.resize(1180, 760)
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()

        interval_ms = int(1000.0 / max(config.ui.refresh_hz, 1.0))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_refresh)
        self._timer.start(interval_ms)

    # ----- construction -------------------------------------------------- #

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        self.estop_btn = QtWidgets.QPushButton("■ EMERGENCY STOP")
        self.estop_btn.setStyleSheet(
            "QPushButton { background:#d83030; color:white; font-weight:700; "
            "padding:4px 14px; border-radius:4px; }"
        )
        self.estop_btn.clicked.connect(self._toggle_estop)
        tb.addWidget(self.estop_btn)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        tb.addWidget(spacer)
        tb.addWidget(QtWidgets.QLabel(f"Profile: {self.config.profile_name}   "))

    def _build_tabs(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.acq_ws = AcquisitionWorkspace(self)
        self.channels_ws = ChannelsWorkspace(self)
        self.raw_ws = RawEEGWorkspace(self)
        self.preprocessing_ws = PreprocessingWorkspace(self)
        self.quality_ws = SignalQualityWorkspace(self)
        self.calibration_ws = CalibrationWorkspace(self)
        self.control_ws = ControlWorkspace(self)
        self.spectral_ws = SpectralWorkspace(self)
        self.replay_ws = ReplayWorkspace(self)
        self.analysis_ws = AnalysisWorkspace(self)
        self.recording_ws = RecordingWorkspace(self)
        self.tabs.addTab(self.acq_ws, "Connection / Acquisition")
        self.tabs.addTab(self.channels_ws, "Channels")
        self.tabs.addTab(self.raw_ws, "Raw EEG")
        self.tabs.addTab(self.preprocessing_ws, "Preprocessing")
        self.tabs.addTab(self.quality_ws, "Signal Quality")
        self.tabs.addTab(self.spectral_ws, "Spectral / State")
        self.tabs.addTab(self.calibration_ws, "Calibration")
        self.tabs.addTab(self.control_ws, "Control / BCI")
        self.tabs.addTab(self.replay_ws, "Replay")
        self.tabs.addTab(self.analysis_ws, "ERP / Epoch Average")
        self.tabs.addTab(self.recording_ws, "Recording")
        self.setCentralWidget(self.tabs)

    def _build_statusbar(self) -> None:
        self.status = StatusBar()
        self.statusBar().addPermanentWidget(self.status, 1)

    # ----- engine control (called by workspaces) ------------------------- #

    def start_acquisition(self, source_type: str, lsl_name: str = "") -> None:
        if self.engine.running:
            return
        self.config.acquisition.source_type = source_type
        self.config.acquisition.lsl_stream_name = lsl_name
        try:
            self.engine.start()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start acquisition.")
            self.engine.stop()
            QtWidgets.QMessageBox.critical(
                self, "Acquisition error",
                f"Could not start the {source_type} source:\n\n{exc}",
            )
            self.acq_ws.set_running(False)
            return
        self.acq_ws.set_running(True)

    def stop_acquisition(self) -> None:
        self.engine.stop()
        self.acq_ws.set_running(False)

    def start_recording(self, participant_id: str, notes: str) -> None:
        if not self.engine.running:
            QtWidgets.QMessageBox.warning(
                self, "Not acquiring",
                "Start acquisition before recording.",
            )
            return
        try:
            self.engine.start_recording(participant_id=participant_id, notes=notes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start recording.")
            QtWidgets.QMessageBox.critical(self, "Recording error", str(exc))

    def stop_recording(self) -> None:
        stats = self.engine.stop_recording()
        if stats is not None:
            QtWidgets.QMessageBox.information(
                self, "Recording saved",
                f"Saved {stats.n_samples:,} samples "
                f"({stats.duration_s:.1f}s, {stats.n_markers} markers).",
            )

    def set_calibration_model(self, model, result) -> None:
        """Store the freshly calibrated model and surface it in the status."""
        self.model = model
        self.calibration_result = result
        if model is not None:
            self.engine.state.update(
                model_name=f"{model.paradigm}/{model.name}",
                paradigm=model.paradigm,
            )

    def _toggle_estop(self) -> None:
        snap = self.engine.state.snapshot()
        new_val = not snap.emergency_stop
        self.engine.state.update(emergency_stop=new_val)
        # In Phase 1 there is no external control yet; the emergency stop
        # latches the safety flag (which gates command output in later
        # phases) and is surfaced everywhere. It does not stop acquisition.
        self.estop_btn.setText("▶ CLEAR E-STOP" if new_val else "■ EMERGENCY STOP")

    # ----- refresh loop -------------------------------------------------- #

    def _on_refresh(self) -> None:
        snap = self.engine.state.snapshot()

        self._tick += 1
        if self._tick % self._quality_every == 0:
            self._last_quality = self._central_quality(snap)

        self.status.update_status(snap, self._last_quality)

        current = self.tabs.currentWidget()
        if hasattr(current, "update_view"):
            try:
                current.update_view()
            except Exception:  # noqa: BLE001 - a view error must not crash the app
                logger.exception("Workspace refresh failed.")

    def _central_quality(self, snap) -> QualityRating | None:
        info = self.engine.stream_info
        if info is None or self.engine.buffer is None:
            return None
        # Deliberately on the RAW stream: this rating gates command execution
        # (safety), and a notch/CAR/clamp stage in the pipeline could mask a
        # genuine electrode fault. Per-tab views can opt into preprocessed.
        data, _ = self.engine.latest_seconds(2.0)
        report = compute_quality(data, info)
        return report.overall_rating if report.channels else None

    # ----- shutdown ------------------------------------------------------ #

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._timer.stop()
        self.engine.stop()
        super().closeEvent(event)
