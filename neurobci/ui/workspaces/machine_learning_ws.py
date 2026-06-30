"""Machine-Learning workspace (JASP-inspired).

Left: data / conditions. Centre: feature set, model selection (optional-deps
gated), preprocessing, cross-validation and chance testing. Right: ranked model
comparison, per-model confusion/ROC plots, diagnostics, report and exports.
Training runs on a background thread with progress and cancel, so the UI never
freezes. Leakage safety is enforced by the evaluation core (all data-dependent
steps fit inside CV folds; every classifier is compared to a chance baseline).

Scope note: classification is complete here; regression and clustering are
provided in a basic form. Deep-learning models, SHAP and temporal
generalization are wired in the core/roadmap and surfaced incrementally.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets

from neurobci.analysis.ml import evaluation as ev
from neurobci.analysis.ml import models as ml_models
from neurobci.analysis.ml import report as ml_report
from neurobci.analysis.ml.features import (MLFeatureConfig, RawEpochsSpec,
                                           extract_features, tensor_X)
from neurobci.analysis.ml.pipelines import PreprocOptions, build_pipeline
from neurobci.analysis.shared.config import AnalysisConfig
from neurobci.analysis.shared.worker import AnalysisWorker
from neurobci.analysis.stats.features import (BandPowerSpec, ERPFeatureSpec,
                                              STANDARD_ROIS)
from neurobci.ui.widgets.condition_selector import ConditionSelector

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

FEATURE_SETS = ["ERP window (per channel)", "Band power (per channel)",
                "Raw epochs (downsampled)", "Raw epochs (tensor, spatial)"]
TENSOR_SET = "Raw epochs (tensor, spatial)"
CV_STRATEGIES = ["stratified_kfold", "repeated_stratified", "kfold", "loo",
                 "group_kfold", "logo"]
CLF_SCORING = ["balanced_accuracy", "accuracy", "roc_auc", "f1_macro"]


class MachineLearningWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._bundle = None
        self._evals: list = []
        self._worker = None
        self._last_config = None
        self._build()

    # ----- construction -------------------------------------------------- #
    def _build(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        split = QtWidgets.QSplitter()

        self.selector = ConditionSelector(self._ctl)
        self.selector.bundleReady.connect(self._on_bundle)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(QtWidgets.QLabel("<b>Data &amp; classes</b>"))
        ll.addWidget(self.selector)
        split.addWidget(left)

        split.addWidget(self._build_center())
        split.addWidget(self._build_right())
        split.setSizes([320, 380, 560])
        root.addWidget(split)

    def _build_center(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(w)
        cl.addWidget(QtWidgets.QLabel("<b>Models &amp; evaluation</b>"))

        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([
            "— preset —",
            "ERP: xDAWN + Logistic Regression",
            "ERP: mean amplitude + LDA",
            "ERP window + SVM",
            "Spectral: band power + Random Forest",
            "Motor imagery: CSP + LDA",
            "Riemannian: tangent space + LR"])
        self.preset.currentIndexChanged.connect(self._apply_preset)
        cl.addWidget(self.preset)

        form = QtWidgets.QFormLayout()
        self.feature_set = QtWidgets.QComboBox()
        self.feature_set.addItems(FEATURE_SETS)
        self.feature_set.currentIndexChanged.connect(self._refresh_models)
        self.tmin = self._dspin(-2, 5, 0.0)
        self.tmax = self._dspin(-2, 5, 0.6)
        self.fmin = self._dspin(0.5, 100, 4.0)
        self.fmax = self._dspin(0.5, 100, 30.0)
        self.roi = QtWidgets.QComboBox()
        self.roi.addItems(["All EEG"] + list(STANDARD_ROIS))
        self.downsample = QtWidgets.QSpinBox()
        self.downsample.setRange(1, 32)
        self.downsample.setValue(4)
        form.addRow("Feature set:", self.feature_set)
        form.addRow("t-min (s):", self.tmin)
        form.addRow("t-max (s):", self.tmax)
        form.addRow("band f-min (Hz):", self.fmin)
        form.addRow("band f-max (Hz):", self.fmax)
        form.addRow("ROI:", self.roi)
        form.addRow("Downsample:", self.downsample)
        cl.addLayout(form)

        cl.addWidget(QtWidgets.QLabel("Models (tick to compare):"))
        self.model_list = QtWidgets.QListWidget()
        self.model_list.setMaximumHeight(180)
        cl.addWidget(self.model_list)

        pp = QtWidgets.QGroupBox("Preprocessing (fit inside CV folds)")
        pf = QtWidgets.QFormLayout(pp)
        self.scaler = QtWidgets.QComboBox()
        self.scaler.addItems(["standard", "robust", "minmax", "none"])
        self.select_k = QtWidgets.QSpinBox()
        self.select_k.setRange(0, 5000)
        self.select_k.setSpecialValueText("off")
        self.class_weight = QtWidgets.QCheckBox("Balance classes (class_weight)")
        pf.addRow("Scaler:", self.scaler)
        pf.addRow("SelectKBest (0=off):", self.select_k)
        pf.addRow(self.class_weight)
        cl.addWidget(pp)

        cvb = QtWidgets.QGroupBox("Cross-validation")
        cf = QtWidgets.QFormLayout(cvb)
        self.cv = QtWidgets.QComboBox()
        self.cv.addItems(CV_STRATEGIES)
        self.n_splits = QtWidgets.QSpinBox()
        self.n_splits.setRange(2, 20)
        self.n_splits.setValue(5)
        self.scoring = QtWidgets.QComboBox()
        self.scoring.addItems(CLF_SCORING)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        self.chance = QtWidgets.QCheckBox("Permutation chance test")
        self.chance.setChecked(True)
        self.n_perm = QtWidgets.QSpinBox()
        self.n_perm.setRange(20, 5000)
        self.n_perm.setValue(100)
        cf.addRow("Strategy:", self.cv)
        cf.addRow("n splits:", self.n_splits)
        cf.addRow("Scoring:", self.scoring)
        cf.addRow("Seed:", self.seed)
        cf.addRow(self.chance)
        cf.addRow("Permutations:", self.n_perm)
        cl.addWidget(cvb)

        run = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Train & compare")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        run.addWidget(self.run_btn)
        run.addWidget(self.cancel_btn)
        cl.addLayout(run)
        self.progress = QtWidgets.QProgressBar()
        cl.addWidget(self.progress)
        cl.addStretch(1)
        self._refresh_models()
        return w

    def _build_right(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(w)
        rl.addWidget(QtWidgets.QLabel("<b>Model comparison</b>"))
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Model", "CV mean", "±", "95% CI", "Chance", "p", "fit s"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select_model)
        rl.addWidget(self.table, 1)

        self.tabs = QtWidgets.QTabWidget()
        if _HAVE_MPL:
            self.fig = Figure(figsize=(4, 3), facecolor="#14181c")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.tabs.addTab(self.canvas, "Plots")
        else:
            self.canvas = None
        self.summary = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.summary, "Report")
        self.diag = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.diag, "Diagnostics")
        rl.addWidget(self.tabs, 1)

        exp = QtWidgets.QHBoxLayout()
        for label, slot in (("Export HTML", lambda: self._export("html")),
                            ("Export CSV", lambda: self._export("csv")),
                            ("Export model (joblib)", self._export_model),
                            ("Save to History", self._save_history)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            exp.addWidget(b)
        rl.addLayout(exp)
        return w

    def _dspin(self, lo, hi, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(3)
        s.setSingleStep(0.05)
        s.setValue(val)
        return s

    # ----- state --------------------------------------------------------- #
    def _on_bundle(self, bundle) -> None:
        self._bundle = bundle
        self.run_btn.setEnabled(True)
        self.summary.setText(
            f"{bundle.n_trials} trials, classes: "
            + ", ".join(f"{k}={v}" for k, v in bundle.class_counts().items())
            + ". Pick models and click Train & compare.")

    def _wants_tensor(self) -> bool:
        return self.feature_set.currentText() == TENSOR_SET

    def _refresh_models(self) -> None:
        data_shape = "tensor" if self._wants_tensor() else "tabular"
        reg = ml_models.registry("classification")
        self.model_list.clear()
        for key, spec in reg.items():
            if spec.data != data_shape:
                continue
            item = QtWidgets.QListWidgetItem(spec.label)
            item.setData(QtCore.Qt.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            if spec.available:
                item.setCheckState(QtCore.Qt.Unchecked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
                item.setFlags(QtCore.Qt.NoItemFlags)
                item.setText(f"{spec.label}  — needs: {spec.install_hint}")
            self.model_list.addItem(item)
        # tick a sensible default
        for i in range(self.model_list.count()):
            it = self.model_list.item(i)
            if it.flags() & QtCore.Qt.ItemIsUserCheckable:
                it.setCheckState(QtCore.Qt.Checked)
                break

    def _apply_preset(self, idx: int) -> None:
        name = self.preset.currentText()
        mapping = {
            "ERP: xDAWN + Logistic Regression": (TENSOR_SET, "xdawn_lr"),
            "ERP: mean amplitude + LDA": (FEATURE_SETS[0], "lda"),
            "ERP window + SVM": (FEATURE_SETS[0], "svm_linear"),
            "Spectral: band power + Random Forest": (FEATURE_SETS[1], "rf"),
            "Motor imagery: CSP + LDA": (TENSOR_SET, "csp_lda"),
            "Riemannian: tangent space + LR": (TENSOR_SET, "riemann_lr"),
        }
        if name not in mapping:
            return
        fset, key = mapping[name]
        self.feature_set.setCurrentText(fset)
        self._refresh_models()
        for i in range(self.model_list.count()):
            it = self.model_list.item(i)
            if it.flags() & QtCore.Qt.ItemIsUserCheckable:
                it.setCheckState(QtCore.Qt.Checked if it.data(QtCore.Qt.UserRole) == key
                                 else QtCore.Qt.Unchecked)

    # ----- features ------------------------------------------------------ #
    def _channels(self):
        roi = self.roi.currentText()
        if roi == "All EEG":
            return None
        present = set(self._bundle.channel_names)
        return [c for c in STANDARD_ROIS[roi] if c in present] or None

    def _features(self):
        """Return ``(X, names)`` for the chosen feature set."""
        chans = self._channels()
        fs = self.feature_set.currentText()
        if fs == TENSOR_SET:
            return tensor_X(self._bundle, chans), ["raw_tensor"]
        if fs == "ERP window (per channel)":
            cfg = MLFeatureConfig([ERPFeatureSpec("mean_amplitude", self.tmin.value(),
                                                  self.tmax.value(), channels=chans)])
        elif fs == "Band power (per channel)":
            cfg = MLFeatureConfig([BandPowerSpec(self.fmin.value(), self.fmax.value(),
                                                 channels=chans)])
        else:
            cfg = MLFeatureConfig([RawEpochsSpec(downsample=self.downsample.value(),
                                                 channels=chans)])
        return extract_features(self._bundle, cfg)

    def _checked_models(self) -> list[str]:
        keys = []
        for i in range(self.model_list.count()):
            it = self.model_list.item(i)
            if (it.flags() & QtCore.Qt.ItemIsUserCheckable) and \
                    it.checkState() == QtCore.Qt.Checked:
                keys.append(it.data(QtCore.Qt.UserRole))
        return keys

    def _opts(self) -> PreprocOptions:
        return PreprocOptions(
            scaler=self.scaler.currentText(),
            select_k=self.select_k.value() or None,
            class_weight_balanced=self.class_weight.isChecked())

    # ----- run (background) ---------------------------------------------- #
    def _run(self) -> None:
        if self._bundle is None:
            return
        if len(self._bundle.condition_names) < 2:
            QtWidgets.QMessageBox.warning(self, "Need 2 classes",
                                          "Select at least two conditions.")
            return
        keys = self._checked_models()
        if not keys:
            QtWidgets.QMessageBox.warning(self, "No models",
                                          "Tick at least one model.")
            return
        try:
            X, _ = self._features()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Feature error", str(exc))
            return
        y = self._bundle.y
        groups = self._bundle.groups
        strategy = self.cv.currentText()
        cv, grouped, cv_desc = ev.make_cv(strategy, n_splits=self.n_splits.value(),
                                          seed=self.seed.value())
        if grouped and len(np.unique(groups)) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Grouped CV needs subjects",
                "Grouped / leave-one-subject-out CV needs >1 session loaded.")
            return
        opts = self._opts()
        scoring = self.scoring.currentText()
        do_chance = self.chance.isChecked()
        n_perm = self.n_perm.value()
        seed = self.seed.value()
        class_names = list(self._bundle.condition_names)

        def job(progress, is_cancelled):
            evals = []
            for i, key in enumerate(keys):
                if is_cancelled():
                    break
                spec = ml_models.get_spec(key)
                progress(100 * i / len(keys), f"Training {spec.label}…")
                pipe = build_pipeline(spec, dict(spec.default_params), opts)
                res = ev.evaluate_classification(
                    pipe, X, y, key=key, label=spec.label, scoring=scoring,
                    cv=cv, is_grouped=grouped, groups=groups if grouped else None,
                    cv_desc=cv_desc, params=dict(spec.default_params),
                    class_names=class_names)
                if do_chance and not is_cancelled():
                    try:
                        ev.add_chance_level(
                            res, pipe, X, y, cv=cv,
                            groups=groups if grouped else None,
                            n_permutations=n_perm, scoring=scoring, seed=seed)
                    except Exception:  # noqa: BLE001
                        logger.exception("Chance test failed for %s", key)
                evals.append(res)
            progress(100, "done")
            return ev.rank_models(evals)

        self._start_worker(job, cv_desc, scoring, opts)

    def _start_worker(self, job, cv_desc, scoring, opts) -> None:
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self._cv_desc = cv_desc
        self._scoring = scoring
        self._opts_used = opts
        self._worker = AnalysisWorker(job, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, pct, msg) -> None:
        self.progress.setValue(pct)
        self.progress.setFormat(f"{msg} %p%")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_failed(self, msg) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Training error", msg)

    def _on_done(self, evals) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._evals = evals
        self._populate_table()
        self._plot_comparison()
        cfg = self._build_config()
        self._last_config = cfg
        report = ml_report.report_for_models(cfg, self._bundle, evals,
                                             figures=None)
        self.summary.setHtml(report.to_html())
        self._last_report = report
        diags = [d for e in evals for d in e.diagnostics
                 if getattr(d, "level", "info") != "info"]
        self.diag.setHtml("<br>".join(
            f'<span style="color:#9a6b00">[{d.level}]</span> {d.message}'
            for d in diags) or "No warnings.")

    def _build_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            kind="ml", analysis_type="classification", seed=self.seed.value(),
            data_source=self.selector.source_info(),
            selection={"conditions": list(self._bundle.condition_names),
                       "roi": self.roi.currentText()},
            features={"families": [self.feature_set.currentText()]},
            params={"cv_desc": self._cv_desc, "scoring": self._scoring,
                    "preproc": f"scaler={self.scaler.currentText()}, "
                               f"k={self.select_k.value() or 'off'}, "
                               f"balance={self.class_weight.isChecked()}",
                    "models": [e.key for e in self._evals]})

    # ----- results ------------------------------------------------------- #
    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._evals))
        for row, e in enumerate(self._evals):
            r = e.summary_row()
            vals = [r["model"], f"{r['cv_mean']:.3f}", f"{r['cv_std']:.3f}",
                    r["ci95"], "—" if r["chance"] is None else f"{r['chance']:.3f}",
                    "—" if r["p_vs_chance"] is None else f"{r['p_vs_chance']:.3g}",
                    f"{r['fit_s']:.2f}"]
            for col, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(v))
                item.setData(QtCore.Qt.UserRole, e.key)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        if self._evals:
            self.table.selectRow(0)

    def _on_select_model(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        key = items[0].data(QtCore.Qt.UserRole)
        e = next((x for x in self._evals if x.key == key), None)
        if e is not None:
            self._plot_model(e)

    # ----- plots --------------------------------------------------------- #
    def _axes(self, n=1):
        self.fig.clear()
        axes = self.fig.subplots(1, n, squeeze=False)[0]
        for ax in axes:
            ax.set_facecolor("#14181c")
            for s in ax.spines.values():
                s.set_color("#5a6672")
            ax.tick_params(colors="#cdd4da", labelsize=8)
            ax.title.set_color("#cdd4da")
            ax.xaxis.label.set_color("#cdd4da")
            ax.yaxis.label.set_color("#cdd4da")
        return axes

    def _plot_comparison(self) -> None:
        if self.canvas is None or not self._evals:
            return
        ax = self._axes(1)[0]
        labels = [e.label for e in self._evals]
        means = [e.mean_score for e in self._evals]
        errs = [e.std_score for e in self._evals]
        ypos = np.arange(len(labels))
        ax.barh(ypos, means, xerr=errs, color="#4f9fe0", alpha=0.8)
        chance = next((e.chance_score for e in self._evals
                       if e.chance_score is not None), None)
        if chance is not None:
            ax.axvline(chance, color="#e0823a", ls="--", lw=1.5, label="chance")
            ax.legend(facecolor="#1b2026", labelcolor="#cdd4da", edgecolor="#5a6672")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(self._scoring)
        ax.set_title("Model comparison (CV)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_model(self, e) -> None:
        if self.canvas is None:
            return
        has_roc = (e.y_proba is not None and len(e.class_labels) == 2)
        axes = self._axes(2 if has_roc else 1)
        cm = e.confusion
        ax0 = axes[0]
        if cm is not None:
            im = ax0.imshow(cm, cmap="Blues")
            ax0.set_xticks(range(len(e.class_labels)))
            ax0.set_yticks(range(len(e.class_labels)))
            ax0.set_xticklabels(e.class_labels, fontsize=8)
            ax0.set_yticklabels(e.class_labels, fontsize=8)
            ax0.set_xlabel("Predicted")
            ax0.set_ylabel("True")
            ax0.set_title(f"{e.label} — confusion")
            for (i, j), v in np.ndenumerate(cm):
                ax0.text(j, i, str(v), ha="center", va="center",
                         color="#ffffff" if v > cm.max() / 2 else "#1c2530",
                         fontsize=8)
        if has_roc:
            from sklearn.metrics import roc_curve
            p1 = e.y_proba[:, 1] if e.y_proba.ndim == 2 else e.y_proba
            classes = np.unique(e.y_true)
            fpr, tpr, _ = roc_curve(e.y_true, p1, pos_label=classes[1])
            ax1 = axes[1]
            ax1.plot(fpr, tpr, color="#36c24a", lw=2)
            ax1.plot([0, 1], [0, 1], ls="--", color="#777777", lw=1)
            ax1.set_xlabel("FPR")
            ax1.set_ylabel("TPR")
            auc = e.metrics.get("roc_auc")
            ax1.set_title(f"ROC (AUC={auc:.3f})" if auc else "ROC")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.tabs.setCurrentIndex(0)

    # ----- export -------------------------------------------------------- #
    def _export(self, kind: str) -> None:
        if not self._evals:
            QtWidgets.QMessageBox.information(self, "Nothing to export",
                                             "Train models first.")
            return
        if kind == "html":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export HTML", "ml_report.html", "HTML (*.html)")
            if path:
                self._last_report.save(path)
        elif kind == "csv":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export CSV", "ml_models.csv", "CSV (*.csv)")
            if path:
                from neurobci.analysis.shared.export import save_table_csv
                rows = [e.summary_row() for e in self._evals]
                cols = {k: [r[k] for r in rows] for k in rows[0]}
                save_table_csv(cols, path)
        if kind in ("html", "csv"):
            QtWidgets.QMessageBox.information(self, "Exported", "Saved.")

    def _export_model(self) -> None:
        if not self._evals or self._bundle is None:
            return
        items = self.table.selectedItems()
        key = items[0].data(QtCore.Qt.UserRole) if items else self._evals[0].key
        spec = ml_models.get_spec(key)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export trained model", f"{key}.joblib", "joblib (*.joblib)")
        if not path:
            return
        try:
            import joblib
            X, _ = self._features()
            pipe = build_pipeline(spec, dict(spec.default_params), self._opts_used)
            pipe.fit(X, self._bundle.y)            # fit on all data for deployment
            joblib.dump({"pipeline": pipe, "classes": self._bundle.condition_names,
                         "config": self._last_config.to_dict()}, path)
            QtWidgets.QMessageBox.information(
                self, "Exported",
                f"Saved model fit on all data (for deployment, not evaluation):\n{path}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def _save_history(self) -> None:
        if not self._evals or self._last_config is None:
            return
        best = self._evals[0]
        self._ctl.analysis_history.add(
            self._last_config,
            summary={"best": best.label, "score": round(best.mean_score, 3),
                     "scoring": best.scoring})
        if hasattr(self._ctl, "history_ws"):
            self._ctl.history_ws.refresh()
        QtWidgets.QMessageBox.information(self, "Saved", "Added to Analysis History.")

    def update_view(self) -> None:
        pass
