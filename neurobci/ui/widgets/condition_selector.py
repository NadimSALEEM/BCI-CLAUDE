"""Reusable condition/epoch selector for the Statistics and ML tabs.

Loads the session currently open in the Replay tab, lets the analyst tick
marker labels and group them into named conditions, sets the epoch window and
preprocessing, and produces a labelled
:class:`~neurobci.analysis.datasource.EpochBundle` via the shared data bridge
(so it sees the same montage and calibrated preprocessing as the ERP tab).
Nothing here touches live acquisition.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets

from neurobci.analysis.datasource import (adopt_montage, bundle_from_session,
                                          offline_pipeline)
from neurobci.bci.erp_analysis import Condition, marker_label_counts
from neurobci.paradigms.base import EpochWindow
from neurobci.recording.external import load_session_any

logger = logging.getLogger(__name__)

_COLS = ["Use", "Label", "Count", "Condition"]


class ConditionSelector(QtWidgets.QWidget):
    """Left-pane data/condition picker. Emits :attr:`bundleReady` on build."""

    bundleReady = QtCore.pyqtSignal(object)      # EpochBundle

    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._session = None
        self._note = ""
        self._build()

    # ----- construction -------------------------------------------------- #
    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.load_btn = QtWidgets.QPushButton("Load from Replay tab")
        self.load_btn.clicked.connect(self._load)
        lay.addWidget(self.load_btn)
        self.path_lbl = QtWidgets.QLabel("No session loaded.")
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setStyleSheet("color:#8fa3b0;")
        lay.addWidget(self.path_lbl)

        tools = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("search event…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        sel_all = QtWidgets.QPushButton("Select all")
        sel_all.clicked.connect(lambda: self._set_all(True))
        unsel = QtWidgets.QPushButton("Unselect all")
        unsel.clicked.connect(lambda: self._set_all(False))
        tools.addWidget(self.search, 1)
        tools.addWidget(sel_all)
        tools.addWidget(unsel)
        lay.addLayout(tools)

        self.table = QtWidgets.QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)

        win = QtWidgets.QGroupBox("Epoch window")
        form = QtWidgets.QFormLayout(win)
        self.tmin = self._dspin(-2, 0, -0.1)
        self.tmax = self._dspin(0.05, 5, 0.6)
        self.baseline = QtWidgets.QCheckBox("Baseline correct")
        self.baseline.setChecked(True)
        self.bmin = self._dspin(-2, 0, -0.1)
        self.bmax = self._dspin(-2, 2, 0.0)
        self.reject = QtWidgets.QCheckBox("Reject by amplitude")
        self.reject.setChecked(True)
        self.reject_uv = self._dspin(10, 1000, 150)
        self.preproc = QtWidgets.QComboBox()
        self.preproc.addItem("Configured pipeline (offline)", True)
        self.preproc.addItem("Raw (no preprocessing)", False)
        form.addRow("tmin (s):", self.tmin)
        form.addRow("tmax (s):", self.tmax)
        form.addRow(self.baseline)
        form.addRow("baseline start:", self.bmin)
        form.addRow("baseline end:", self.bmax)
        form.addRow(self.reject)
        form.addRow("reject (uV):", self.reject_uv)
        form.addRow("Preprocessing:", self.preproc)
        lay.addWidget(win)

        self.build_btn = QtWidgets.QPushButton("Epoch ticked conditions →")
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self._emit_bundle)
        lay.addWidget(self.build_btn)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def _dspin(self, lo, hi, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(3)
        s.setSingleStep(0.05)
        s.setValue(val)
        return s

    # ----- loading ------------------------------------------------------- #
    def _load(self) -> None:
        path = getattr(self._ctl.replay_ws, "_session_path", None)
        if not path:
            QtWidgets.QMessageBox.information(
                self, "No replay session",
                "Load a session in the Replay tab first.")
            return
        try:
            session = load_session_any(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load failed.")
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        session, self._note = adopt_montage(session, self._ctl.engine,
                                            self._ctl.config)
        self._session = session
        self.path_lbl.setText(f"{path}\n{self._note}")
        self._populate()
        self.build_btn.setEnabled(True)

    def _populate(self) -> None:
        counts = marker_label_counts(self._session.markers)
        self.table.setRowCount(len(counts))
        for row, (label, count) in enumerate(counts.items()):
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(row, 0, chk)
            lab = QtWidgets.QTableWidgetItem(label)
            lab.setFlags(QtCore.Qt.ItemIsEnabled)
            lab.setData(QtCore.Qt.UserRole, label)
            self.table.setItem(row, 1, lab)
            cnt = QtWidgets.QTableWidgetItem(str(count))
            cnt.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row, 2, cnt)
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(label))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self._filter(self.search.text())

    def _filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for row in range(self.table.rowCount()):
            label = (self.table.item(row, 1).data(QtCore.Qt.UserRole) or "")
            self.table.setRowHidden(row, needle not in label.lower())

    def _set_all(self, checked: bool) -> None:
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(state)

    # ----- bundle build -------------------------------------------------- #
    def _conditions(self) -> list[Condition]:
        grouped: dict[str, list[str]] = {}
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != QtCore.Qt.Checked:
                continue
            label = self.table.item(row, 1).data(QtCore.Qt.UserRole)
            cond = (self.table.item(row, 3).text() or label).strip()
            grouped.setdefault(cond, []).append(label)
        return [Condition(name=n, labels=labs) for n, labs in grouped.items()]

    def _window(self) -> EpochWindow:
        baseline = (self.bmin.value(), self.bmax.value()) \
            if self.baseline.isChecked() else None
        return EpochWindow(tmin=self.tmin.value(), tmax=self.tmax.value(),
                           baseline=baseline, reject_uv=self.reject_uv.value())

    def build_bundle(self):
        """Epoch the ticked conditions; returns an EpochBundle (or ``None``)."""
        if self._session is None:
            return None
        conditions = self._conditions()
        if len(conditions) < 1:
            QtWidgets.QMessageBox.warning(self, "No conditions",
                                          "Tick at least one marker label.")
            return None
        window = self._window()
        pipe, note = offline_pipeline(
            self._session, self._ctl.config, self._ctl.engine,
            use_preprocessing=bool(self.preproc.currentData()))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            bundle = bundle_from_session(
                self._session, conditions, window,
                preprocess=pipe, reject=self.reject.isChecked())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Epoching failed.")
            QtWidgets.QMessageBox.critical(self, "Epoching error", str(exc))
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        counts = ", ".join(f"{k}={v}" for k, v in bundle.class_counts().items())
        self.status.setText(f"{bundle.n_trials} trials ({counts}) · {note}")
        return bundle

    def _emit_bundle(self) -> None:
        bundle = self.build_bundle()
        if bundle is not None:
            self.bundleReady.emit(bundle)

    def source_info(self) -> dict:
        """Reproducibility snapshot of the data source."""
        if self._session is None:
            return {}
        return {"path": getattr(self._ctl.replay_ws, "_session_path", ""),
                "sfreq": self._session.sfreq,
                "n_channels": len(self._session.channel_names),
                "montage_note": self._note,
                "window": {"tmin": self.tmin.value(), "tmax": self.tmax.value(),
                           "baseline": self.baseline.isChecked(),
                           "reject_uv": self.reject_uv.value()},
                "preprocessing": bool(self.preproc.currentData())}
