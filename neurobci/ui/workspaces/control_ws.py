"""Online BCI control demonstration workspace.

A safe, internal on-screen task: items sit on a ring and a cursor moves
toward the item the BCI selects. Each selection runs the *real* control
stack -- SelectionController (early-stopping P300 decision) -> SafetyMonitor
-> CommandRouter -> internal adapter -> history + metrics. Only the
stimulus source is simulated.

Safety is front-and-centre: nothing executes while the emergency stop is
active, the stream is disconnected, signal quality is bad, or test mode is
on -- and the reason is shown in the history.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from neurobci.control.adapters import BoardModel, InternalBoardAdapter
from neurobci.control.router import CommandRouter
from neurobci.control.simulated_driver import run_selection_trial
from neurobci.core.app_state import ConnectionStatus
from neurobci.quality.metrics import QualityRating

logger = logging.getLogger(__name__)

_CONNECTED = {ConnectionStatus.CONNECTED, ConnectionStatus.SIMULATED,
              ConnectionStatus.REPLAYED}


class ControlWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._router: CommandRouter | None = None
        self._board: BoardModel | None = None
        self._item_scatter = None
        self._cursor_scatter = None
        self._trail = None
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QHBoxLayout(self)

        # --- left: board ------------------------------------------------- #
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setXRange(-1.4, 1.4)
        self.plot.setYRange(-1.4, 1.4)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        root.addWidget(self.plot, stretch=3)

        # --- right: controls -------------------------------------------- #
        side = QtWidgets.QVBoxLayout()

        self.model_label = QtWidgets.QLabel("No model — calibrate first.")
        self.model_label.setWordWrap(True)
        side.addWidget(self.model_label)

        form = QtWidgets.QFormLayout()
        self.intended = QtWidgets.QComboBox()
        self.test_mode = QtWidgets.QCheckBox("Test mode (predict, don't act)")
        self.test_mode.stateChanged.connect(self._on_test_mode)
        form.addRow("Intended item:", self.intended)
        form.addRow("", self.test_mode)
        side.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run selection")
        self.run_btn.clicked.connect(lambda: self._run_one())
        self.auto_btn = QtWidgets.QPushButton("Run 6 trials")
        self.auto_btn.clicked.connect(self._run_auto)
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.auto_btn)
        btns.addWidget(self.reset_btn)
        side.addLayout(btns)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        side.addWidget(self.status)

        self.metrics_label = QtWidgets.QLabel("")
        self.metrics_label.setStyleSheet("font-family: monospace;")
        side.addWidget(self.metrics_label)

        side.addWidget(QtWidgets.QLabel("Command history:"))
        self.history = QtWidgets.QListWidget()
        side.addWidget(self.history, stretch=1)

        root.addLayout(side, stretch=2)

    # ----- board / router setup ----------------------------------------- #

    def _ensure(self) -> None:
        n = self._ctl.config.control.n_items
        if self._router is not None and self._board is not None and self._board.n_items == n:
            return
        self._board = BoardModel(n_items=n)
        self._router = CommandRouter(InternalBoardAdapter(self._board), self._ctl.config.control)
        self.intended.clear()
        self.intended.addItems([str(i) for i in range(n)])
        self._build_board_items()

    def _build_board_items(self) -> None:
        self.plot.clear()
        xy = self._board.item_xy
        self._item_scatter = pg.ScatterPlotItem(
            xy[:, 0], xy[:, 1], size=34, brush=pg.mkBrush("#26323c"),
            pen=pg.mkPen("#3a4a57"),
        )
        self.plot.addItem(self._item_scatter)
        for i, (x, y) in enumerate(xy):
            t = pg.TextItem(str(i), anchor=(0.5, 0.5), color="#cdd4da")
            t.setPos(x, y)
            self.plot.addItem(t)
        self._trail = self.plot.plot(pen=pg.mkPen("#3a9bd0", width=1))
        self._cursor_scatter = pg.ScatterPlotItem(
            [0], [0], size=22, brush=pg.mkBrush("#ffd070"), pen=pg.mkPen("#ffffff")
        )
        self.plot.addItem(self._cursor_scatter)

    # ----- safety context ------------------------------------------------ #

    def _context(self):
        snap = self._ctl.engine.state.snapshot()
        connected = self._ctl.engine.running and snap.connection in _CONNECTED
        q = self._ctl._last_quality
        quality_ok = None
        if q is not None:
            quality_ok = q in (QualityRating.GOOD, QualityRating.FAIR)
        return snap.emergency_stop, connected, quality_ok

    # ----- run ----------------------------------------------------------- #

    def _on_test_mode(self) -> None:
        self._ctl.config.control.test_mode = self.test_mode.isChecked()

    def _run_one(self, intended: int | None = None) -> None:
        self._ensure()
        model = self._ctl.model
        if model is None:
            QtWidgets.QMessageBox.information(
                self, "No model", "Calibrate a model first (Calibration tab).")
            return
        if model.paradigm != "p300":
            QtWidgets.QMessageBox.information(
                self, "Unsupported",
                "This on-screen selection demo currently supports P300 models. "
                f"The loaded model is '{model.paradigm}'.")
            return
        if intended is None:
            intended = int(self.intended.currentText())
        estop, connected, quality_ok = self._context()

        try:
            outcome, rec = run_selection_trial(
                model, self._router, self._ctl.config.control.selection,
                intended_item=intended, channels=self._ctl.config.channels,
                sfreq=self._ctl.config.acquisition.expected_sfreq,
                emergency_stop=estop, connected=connected, quality_ok=quality_ok,
                seed=np.random.randint(0, 1_000_000),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Selection trial failed.")
            QtWidgets.QMessageBox.critical(self, "Control error", str(exc))
            return

        self._show_outcome(outcome, rec, intended)
        self._update_board()

    def _run_auto(self) -> None:
        self._ensure()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        try:
            for _ in range(self._board.n_items):
                self._run_one(int(np.random.randint(0, self._board.n_items)))
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _reset(self) -> None:
        self._router = None
        self._board = None
        self.history.clear()
        self.status.setText("")
        self.metrics_label.setText("")
        self._ensure()
        self._update_board()

    # ----- views --------------------------------------------------------- #

    def _show_outcome(self, outcome, rec, intended) -> None:
        if outcome.decided:
            verb = "executed" if rec.executed else "BLOCKED"
            mark = "✓" if rec.correct else "✗"
            line = (f"{mark} select {outcome.item} (conf {outcome.confidence:.2f}, "
                    f"margin {outcome.margin:.2f}, {outcome.n_flashes} flashes) "
                    f"[intended {intended}] — {verb}")
            if not rec.executed:
                line += f": {rec.rejected_reason}"
        else:
            line = (f"— abstained (conf {outcome.confidence:.2f}); "
                    f"{outcome.reason}")
        self.history.insertItem(0, line)
        self.status.setText(line)
        self.metrics_label.setText(self._router.metrics.summary())

    def _update_board(self) -> None:
        if self._board is None or self._cursor_scatter is None:
            return
        self._cursor_scatter.setData([self._board.cursor[0]], [self._board.cursor[1]])
        if self._board.trail:
            arr = np.array(self._board.trail)
            self._trail.setData(arr[:, 0], arr[:, 1])

    def update_view(self) -> None:
        self._ensure()
        model = self._ctl.model
        if model is None:
            self.model_label.setText("No model — calibrate first (Calibration tab).")
            self.run_btn.setEnabled(False)
            self.auto_btn.setEnabled(False)
        else:
            self.model_label.setText(
                f"Model: {model.paradigm}/{model.name}  "
                f"(bal.acc {model.metrics.get('balanced_accuracy', float('nan')):.2f})"
            )
            self.run_btn.setEnabled(True)
            self.auto_btn.setEnabled(True)
        self._update_board()
