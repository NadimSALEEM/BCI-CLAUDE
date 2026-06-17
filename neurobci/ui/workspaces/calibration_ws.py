"""Calibration & model-evaluation workspace (P300).

Runs a *simulated* short calibration (synthetic P300 epochs with known
ground truth), compares candidate models with leakage-free cross-
validation, shows the metrics against chance, lets you save the best model,
and offers a quick simulated online-selection verification.

Honesty is built in: if the best model does not clearly beat chance, the
workspace says so in red and refuses to pretend the calibration succeeded.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from neurobci.bci.calibration import run_simulated_calibration
from neurobci.bci.model_store import save_model
from neurobci.bci.online import OnlineP300Decoder
from neurobci.paradigms.base import available_paradigms, get_paradigm
from neurobci.paradigms.synthetic import make_p300_selection_run

logger = logging.getLogger(__name__)

_METRIC_COLS = ["Model", "Bal.acc", "±std", "AUC", "kappa", "MCC", "> chance?"]


class CalibrationWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._result = None
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # --- parameters -------------------------------------------------- #
        params = QtWidgets.QGroupBox("Simulated calibration")
        form = QtWidgets.QFormLayout(params)

        self.paradigm_combo = QtWidgets.QComboBox()
        for name in available_paradigms():
            self.paradigm_combo.addItem(f"{name}  ({get_paradigm(name).family})", name)
        self.paradigm_combo.currentIndexChanged.connect(self._on_paradigm)

        self.n_trials = QtWidgets.QSpinBox()
        self.n_trials.setRange(48, 2000)
        self.n_trials.setValue(300)
        self.amp = QtWidgets.QDoubleSpinBox()
        self.amp.setRange(0.0, 30.0)
        self.amp.setValue(6.0)
        self.amp.setSuffix(" uV signal")
        self.noise = QtWidgets.QDoubleSpinBox()
        self.noise.setRange(0.5, 30.0)
        self.noise.setValue(4.0)
        self.noise.setSuffix(" uV noise")
        self.folds = QtWidgets.QSpinBox()
        self.folds.setRange(2, 10)
        self.folds.setValue(5)

        self._model_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._models_widget = QtWidgets.QWidget()
        self._models_layout = QtWidgets.QHBoxLayout(self._models_widget)
        self._models_layout.setContentsMargins(0, 0, 0, 0)

        form.addRow("Paradigm:", self.paradigm_combo)
        form.addRow("Trials:", self.n_trials)
        form.addRow("Signal amplitude:", self.amp)
        form.addRow("Background noise:", self.noise)
        form.addRow("CV folds:", self.folds)
        form.addRow("Models:", self._models_widget)
        root.addWidget(params)
        self._rebuild_models()

        # --- actions ----------------------------------------------------- #
        actions = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run simulated calibration")
        self.run_btn.clicked.connect(self._run)
        self.save_btn = QtWidgets.QPushButton("Save best model")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.verify_btn = QtWidgets.QPushButton("Online verification (sim)")
        self.verify_btn.setEnabled(False)
        self.verify_btn.clicked.connect(self._verify)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.verify_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        # --- results ----------------------------------------------------- #
        self.table = QtWidgets.QTableWidget(0, len(_METRIC_COLS))
        self.table.setHorizontalHeaderLabels(_METRIC_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table)

        self.summary = QtWidgets.QLabel("No calibration yet.")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        self.confusion = QtWidgets.QLabel("")
        self.confusion.setStyleSheet("font-family: monospace;")
        root.addWidget(self.confusion)
        root.addStretch(1)

    def _wrap(self, layout):
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    def _current_paradigm(self):
        return get_paradigm(self.paradigm_combo.currentData())

    def _on_paradigm(self) -> None:
        self._rebuild_models()

    def _rebuild_models(self) -> None:
        while self._models_layout.count():
            item = self._models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._model_checks = {}
        paradigm = self._current_paradigm()
        names = paradigm.model_names
        for i, name in enumerate(names):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(i == 0)
            cb.setEnabled(name != "cca")     # CCA is the only SSVEP option
            self._model_checks[name] = cb
            self._models_layout.addWidget(cb)
        self._models_layout.addStretch(1)

    # ----- run ----------------------------------------------------------- #

    def _run(self) -> None:
        paradigm = self._current_paradigm()
        models = [n for n, cb in self._model_checks.items() if cb.isChecked()]
        if not models:
            QtWidgets.QMessageBox.warning(self, "No model", "Select at least one model.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.run_btn.setEnabled(False)
        self.summary.setText("Running calibration…")
        QtWidgets.QApplication.processEvents()
        try:
            result = run_simulated_calibration(
                paradigm,
                self._ctl.config.channels,
                sfreq=self._ctl.config.acquisition.expected_sfreq,
                n_trials=self.n_trials.value(),
                p300_amp_uv=self.amp.value(),
                noise_uv=self.noise.value(),
                model_names=models,
                n_folds=self.folds.value(),
                seed=0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Calibration failed.")
            QtWidgets.QMessageBox.critical(self, "Calibration error", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)

        self._result = result
        self._show(result)
        self._ctl.set_calibration_model(result.model, result)

    def _show(self, result) -> None:
        self.table.setRowCount(len(result.results))
        for row, (name, r) in enumerate(result.results.items()):
            beats = "YES" if r.beats_chance else "no"
            vals = [name, f"{r.balanced_accuracy:.3f}", f"{r.balanced_accuracy_std:.3f}",
                    f"{r.roc_auc:.3f}", f"{r.kappa:.3f}", f"{r.mcc:.3f}", beats]
            for col, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                if name == result.best_model_name:
                    item.setBackground(QtGui.QColor("#1f3a5f"))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

        usable = result.is_usable
        self.save_btn.setEnabled(usable)
        # Online verification is the P300 selection demo; only enable for P300.
        self.verify_btn.setEnabled(usable and result.paradigm == "p300")
        counts = ", ".join(f"{k}:{v}" for k, v in result.class_counts.items())
        head = (
            f"Best: <b>{result.best_model_name}</b> &nbsp; "
            f"epochs={result.n_epochs} (classes {counts})"
        )
        if usable:
            self.summary.setText(
                f'<span style="color:#36c24a">✅ {head} — above chance, '
                f"ready to save.</span>"
            )
        else:
            msg = " ".join(result.messages) or "Not above chance."
            self.summary.setText(
                f'<span style="color:#ff6b6b">❌ {head}<br>{msg}</span>'
            )

        if result.best_model_name and result.best_model_name in result.results:
            cm = result.results[result.best_model_name].confusion
            rows = "\n".join(f"  {row}" for row in cm)
            self.confusion.setText("Confusion (rows=true, cols=pred):\n" + rows)

    # ----- save / verify ------------------------------------------------- #

    def _save(self) -> None:
        if self._result is None or self._result.model is None:
            return
        try:
            path = save_model(
                self._result.model, root="models",
                extra_meta={"calibration": "simulated",
                            "n_trials": self.n_trials.value()},
            )
            QtWidgets.QMessageBox.information(self, "Model saved", f"Saved to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Save failed.")
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))

    def _verify(self) -> None:
        if self._result is None or self._result.model is None:
            return
        decoder = OnlineP300Decoder(self._result.model)
        n_runs, correct = 12, 0
        rng = np.random.default_rng(0)
        for k in range(n_runs):
            target = int(rng.integers(0, 6))
            ds, item_ids, true_item = make_p300_selection_run(
                n_items=6, n_repetitions=10, target_item=target,
                channels=self._ctl.config.channels,
                sfreq=self._ctl.config.acquisition.expected_sfreq,
                p300_amp_uv=self.amp.value(), noise_uv=self.noise.value(),
                window=get_paradigm("p300").window, seed=100 + k,
            )
            selected, _ = decoder.decide_selection(ds.X, item_ids, 6)
            correct += int(selected == true_item)
        QtWidgets.QMessageBox.information(
            self, "Online verification",
            f"Simulated selection accuracy: {correct}/{n_runs} "
            f"({100*correct/n_runs:.0f}%).",
        )

    def update_view(self) -> None:
        # Calibration is on-demand; nothing to refresh continuously.
        pass
