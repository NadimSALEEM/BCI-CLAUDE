"""Statistics workspace (JASP-inspired).

Left: data / conditions (shared :class:`ConditionSelector`). Centre: feature and
test settings with presets and a basic/advanced split. Right: results, plots,
diagnostics and a self-contained report. Operates on epochs built from the
Replay session; never touches live acquisition.

Scope note: this is the first, broad-but-not-exhaustive slice of the statistics
spec -- univariate parametric/non-parametric/robust tests, correlation, and
1-D cluster-permutation over time. RM/mixed ANOVA, spatio-temporal clusters and
time-frequency stats are wired in the headless core and exposed incrementally.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets

from neurobci.analysis.shared.config import AnalysisConfig
from neurobci.analysis.shared.deps import MissingDependencyError, have
from neurobci.analysis.shared.validation import Diagnostic
from neurobci.analysis.stats import anova as stats_anova
from neurobci.analysis.stats import mass_univariate as mu
from neurobci.analysis.stats import report as stats_report
from neurobci.analysis.stats import tests as stats_tests
from neurobci.analysis.stats.features import (BandPowerSpec, ERPFeatureSpec,
                                              STANDARD_ROIS, compute_feature,
                                              feature_by_condition)
from neurobci.ui.widgets.condition_selector import ConditionSelector

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

_PALETTE = ["#4f9fe0", "#e0823a", "#36c24a", "#d8508a", "#b07cf0", "#e0c040"]

MODES = ["Two-group test", "Multi-group (ANOVA / Kruskal)",
         "ANOVA / RM / mixed", "Correlation", "Cluster permutation (time)"]
ANOVA_TYPES = ["One-way", "Factorial (two-way)", "Repeated-measures",
               "Friedman (non-parametric RM)", "ANCOVA"]
TWO_GROUP_TESTS = ["Paired t-test", "Independent t-test", "Welch t-test",
                   "Wilcoxon signed-rank", "Mann-Whitney U", "Permutation t-test"]
FEATURES = ["mean_amplitude", "peak_amplitude", "peak_latency", "auc", "gfp",
            "band_power"]


class StatisticsWorkspace(QtWidgets.QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self._ctl = controller
        self._bundle = None
        self._last = None                 # (config, report) for export
        self._figs: list = []
        self._build()

    # ----- construction -------------------------------------------------- #
    def _build(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        split = QtWidgets.QSplitter()

        # left: data / conditions
        self.selector = ConditionSelector(self._ctl)
        self.selector.bundleReady.connect(self._on_bundle)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(QtWidgets.QLabel("<b>Data &amp; conditions</b>"))
        ll.addWidget(self.selector)
        split.addWidget(left)

        # centre: analysis settings
        split.addWidget(self._build_center())

        # right: results
        split.addWidget(self._build_right())
        split.setSizes([330, 360, 560])
        root.addWidget(split)

    def _build_center(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(w)
        cl.addWidget(QtWidgets.QLabel("<b>Analysis</b>"))

        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([
            "— preset —",
            "ERP: condition comparison (mean amplitude)",
            "ERP: paired comparison",
            "ERP: cluster permutation",
            "Spectral: band-power comparison",
            "Correlation: feature vs reaction time",
            "Repeated-measures ANOVA",
            "Mixed / factorial ANOVA"])
        self.preset.currentIndexChanged.connect(self._apply_preset)
        cl.addWidget(self.preset)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.mode.currentIndexChanged.connect(self._refresh_enabled)
        cl.addWidget(self._row("Mode:", self.mode))

        feat = QtWidgets.QGroupBox("Feature")
        ff = QtWidgets.QFormLayout(feat)
        self.feature = QtWidgets.QComboBox()
        self.feature.addItems(FEATURES)
        self.feature.currentIndexChanged.connect(self._refresh_enabled)
        self.tmin = self._dspin(-2, 5, 0.25)
        self.tmax = self._dspin(-2, 5, 0.45)
        self.fmin = self._dspin(0.5, 100, 8.0)
        self.fmax = self._dspin(0.5, 100, 13.0)
        self.peak_sign = QtWidgets.QComboBox()
        self.peak_sign.addItems(["abs", "pos", "neg"])
        self.roi = QtWidgets.QComboBox()
        self.roi.addItems(["All EEG"] + list(STANDARD_ROIS) + ["Single channel"])
        self.roi.currentIndexChanged.connect(self._refresh_enabled)
        self.channel = QtWidgets.QComboBox()
        ff.addRow("Feature:", self.feature)
        ff.addRow("t-min (s):", self.tmin)
        ff.addRow("t-max (s):", self.tmax)
        ff.addRow("f-min (Hz):", self.fmin)
        ff.addRow("f-max (Hz):", self.fmax)
        ff.addRow("Peak sign:", self.peak_sign)
        ff.addRow("ROI:", self.roi)
        ff.addRow("Channel:", self.channel)
        cl.addWidget(feat)

        grp = QtWidgets.QGroupBox("Test & groups")
        gf = QtWidgets.QFormLayout(grp)
        self.test = QtWidgets.QComboBox()
        self.test.addItems(TWO_GROUP_TESTS)
        self.anova_type = QtWidgets.QComboBox()
        self.anova_type.addItems(ANOVA_TYPES)
        self.anova_type.currentIndexChanged.connect(self._refresh_enabled)
        self.subject_unit = QtWidgets.QComboBox()
        self.posthoc = QtWidgets.QCheckBox("Post-hoc pairwise")
        self.posthoc.setChecked(True)
        self.cond_a = QtWidgets.QComboBox()
        self.cond_b = QtWidgets.QComboBox()
        self.metavar = QtWidgets.QComboBox()
        self.alpha = self._dspin(0.0001, 0.5, 0.05)
        self.alternative = QtWidgets.QComboBox()
        self.alternative.addItems(["two-sided", "greater", "less"])
        self.correction = QtWidgets.QComboBox()
        self.correction.addItems(["fdr_bh", "fdr_by", "holm", "bonferroni", "none"])
        self.nperm = QtWidgets.QSpinBox()
        self.nperm.setRange(100, 100000)
        self.nperm.setValue(2000)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        gf.addRow("Test:", self.test)
        gf.addRow("ANOVA type:", self.anova_type)
        gf.addRow("Subject unit:", self.subject_unit)
        gf.addRow(self.posthoc)
        gf.addRow("Condition A:", self.cond_a)
        gf.addRow("Condition B:", self.cond_b)
        gf.addRow("2nd factor / covar / correlate:", self.metavar)
        gf.addRow("alpha:", self.alpha)
        gf.addRow("Alternative:", self.alternative)
        gf.addRow("Correction (map):", self.correction)
        gf.addRow("Permutations:", self.nperm)
        gf.addRow("Random seed:", self.seed)
        cl.addWidget(grp)

        self.run_btn = QtWidgets.QPushButton("Run analysis")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        cl.addWidget(self.run_btn)
        cl.addStretch(1)
        self._refresh_enabled()
        return w

    def _build_right(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(w)
        rl.addWidget(QtWidgets.QLabel("<b>Results</b>"))
        self.tabs = QtWidgets.QTabWidget()
        self.summary = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.summary, "Summary")
        if _HAVE_MPL:
            self.fig = Figure(figsize=(4, 3), facecolor="#14181c")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.tabs.addTab(self.canvas, "Plot")
        else:
            self.canvas = None
        self.diag = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.diag, "Diagnostics")
        rl.addWidget(self.tabs, 1)

        exp = QtWidgets.QHBoxLayout()
        for label, slot in (("Export HTML", lambda: self._export("html")),
                            ("Export CSV", lambda: self._export("csv")),
                            ("Export JSON", lambda: self._export("json")),
                            ("Export PNG", lambda: self._export("png")),
                            ("Save to History", self._save_history)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            exp.addWidget(b)
        rl.addLayout(exp)
        return w

    def _row(self, label, widget):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel(label))
        h.addWidget(widget, 1)
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
        for combo in (self.cond_a, self.cond_b):
            combo.clear()
            combo.addItems(bundle.condition_names)
        if self.cond_b.count() > 1:
            self.cond_b.setCurrentIndex(1)
        self.channel.clear()
        self.channel.addItems(bundle.channel_names)
        meta_keys = [k for k in bundle.metadata if k not in ("onset_sample",)]
        self.metavar.clear()
        self.metavar.addItems(meta_keys)
        self.subject_unit.clear()
        self.subject_unit.addItem("Auto (session)", None)
        for k in meta_keys:
            self.subject_unit.addItem(f"metadata: {k}", k)
        n_subj = len(np.unique(bundle.groups))
        self._n_subjects = n_subj
        self.run_btn.setEnabled(True)
        self.summary.setText(
            f"Epoched {bundle.n_trials} trials across "
            f"{len(bundle.condition_names)} condition(s). Configure a test and "
            f"click Run.")

    def _refresh_enabled(self) -> None:
        mode = self.mode.currentText()
        is_band = self.feature.currentText() == "band_power"
        for w in (self.tmin, self.tmax):
            w.setEnabled(not is_band)
        for w in (self.fmin, self.fmax):
            w.setEnabled(is_band)
        self.peak_sign.setEnabled(self.feature.currentText() in
                                  ("peak_amplitude", "peak_latency"))
        is_anova = mode == "ANOVA / RM / mixed"
        atype = self.anova_type.currentText()
        self.channel.setEnabled(self.roi.currentText() == "Single channel")
        self.test.setEnabled(mode == "Two-group test")
        self.anova_type.setEnabled(is_anova)
        self.subject_unit.setEnabled(is_anova and atype in
                                     ("Repeated-measures", "Friedman (non-parametric RM)"))
        self.posthoc.setEnabled(is_anova)
        self.cond_a.setEnabled(mode in ("Two-group test", "Cluster permutation (time)"))
        self.cond_b.setEnabled(mode in ("Two-group test", "Cluster permutation (time)"))
        # 2nd factor / covariate / correlate variable
        self.metavar.setEnabled(mode == "Correlation" or
                                (is_anova and atype in ("Factorial (two-way)", "ANCOVA")))
        self.correction.setEnabled(mode == "Cluster permutation (time)" or
                                   (is_anova and self.posthoc.isChecked()))
        self.nperm.setEnabled(mode in ("Two-group test", "Cluster permutation (time)"))

    def _apply_preset(self, idx: int) -> None:
        name = self.preset.currentText()
        if name.startswith("ERP: condition comparison"):
            self.mode.setCurrentText("Two-group test")
            self.feature.setCurrentText("mean_amplitude")
            self.test.setCurrentText("Independent t-test")
            self.tmin.setValue(0.25); self.tmax.setValue(0.45)
        elif name.startswith("ERP: paired"):
            self.mode.setCurrentText("Two-group test")
            self.test.setCurrentText("Paired t-test")
        elif name.startswith("ERP: cluster"):
            self.mode.setCurrentText("Cluster permutation (time)")
            self.feature.setCurrentText("mean_amplitude")
        elif name.startswith("Spectral"):
            self.mode.setCurrentText("Two-group test")
            self.feature.setCurrentText("band_power")
            self.fmin.setValue(8.0); self.fmax.setValue(13.0)
        elif name.startswith("Correlation"):
            self.mode.setCurrentText("Correlation")
        elif name.startswith("Repeated-measures"):
            self.mode.setCurrentText("ANOVA / RM / mixed")
            self.anova_type.setCurrentText("Repeated-measures")
        elif name.startswith("Mixed / factorial"):
            self.mode.setCurrentText("ANOVA / RM / mixed")
            self.anova_type.setCurrentText("Factorial (two-way)")
        self._refresh_enabled()

    # ----- feature spec -------------------------------------------------- #
    def _channels(self):
        roi = self.roi.currentText()
        if roi == "All EEG":
            return None
        if roi == "Single channel":
            return [self.channel.currentText()]
        present = set(self._bundle.channel_names)
        return [c for c in STANDARD_ROIS[roi] if c in present] or None

    def _feature_spec(self):
        chans = self._channels()
        if self.feature.currentText() == "band_power":
            return BandPowerSpec(self.fmin.value(), self.fmax.value(), channels=chans)
        return ERPFeatureSpec(self.feature.currentText(), self.tmin.value(),
                              self.tmax.value(), channels=chans,
                              peak_sign=self.peak_sign.currentText())

    def _config(self, analysis_type: str) -> AnalysisConfig:
        spec = self._feature_spec()
        return AnalysisConfig(
            kind="stats", analysis_type=analysis_type, seed=self.seed.value(),
            data_source=self.selector.source_info(),
            selection={"conditions": self._bundle.condition_names,
                       "channels": self._channels() or "all EEG"},
            features={"feature": getattr(spec, "feature", spec.label()),
                      "window": [self.tmin.value(), self.tmax.value()],
                      "band": [self.fmin.value(), self.fmax.value()]},
            params={"test": self.test.currentText(), "alpha": self.alpha.value(),
                    "alternative": self.alternative.currentText(),
                    "correction": self.correction.currentText(),
                    "permutations": self.nperm.value()})

    # ----- run ----------------------------------------------------------- #
    def _run(self) -> None:
        if self._bundle is None:
            return
        try:
            mode = self.mode.currentText()
            if mode == "Two-group test":
                self._run_two_group()
            elif mode == "Multi-group (ANOVA / Kruskal)":
                self._run_multi_group()
            elif mode == "ANOVA / RM / mixed":
                self._run_anova()
            elif mode == "Correlation":
                self._run_correlation()
            else:
                self._run_cluster()
        except MissingDependencyError as exc:
            QtWidgets.QMessageBox.warning(self, "Optional dependency needed", str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Statistics run failed.")
            QtWidgets.QMessageBox.critical(self, "Analysis error", str(exc))

    def _run_two_group(self) -> None:
        spec = self._feature_spec()
        vals = compute_feature(self._bundle, spec)
        by = feature_by_condition(vals, self._bundle)
        a_name, b_name = self.cond_a.currentText(), self.cond_b.currentText()
        if a_name == b_name:
            QtWidgets.QMessageBox.warning(self, "Pick two conditions",
                                          "Condition A and B must differ.")
            return
        a, b = by[a_name], by[b_name]
        test = self.test.currentText()
        alt, alpha = self.alternative.currentText(), self.alpha.value()
        if test == "Paired t-test":
            res = stats_tests.paired_t(a, b, alternative=alt, alpha=alpha)
        elif test == "Independent t-test":
            res = stats_tests.independent_t(a, b, alternative=alt, alpha=alpha)
        elif test == "Welch t-test":
            res = stats_tests.independent_t(a, b, welch=True, alternative=alt, alpha=alpha)
        elif test == "Wilcoxon signed-rank":
            res = stats_tests.wilcoxon(a, b, alternative=alt, alpha=alpha)
        elif test == "Mann-Whitney U":
            res = stats_tests.mann_whitney(a, b, alternative=alt, alpha=alpha)
        else:
            res = stats_tests.permutation_ttest(
                a, b, paired=("Paired" in test), n_permutations=self.nperm.value(),
                alternative=alt, seed=self.seed.value(), alpha=alpha)
        sug, why = stats_tests.assumptions_suggest_nonparametric(res.assumptions, alpha)
        if sug:
            res.diagnostics.append(Diagnostic(
                "warning", f"Assumptions questionable ({why}); consider a "
                f"non-parametric test. Not switched automatically."))
        self._plot_distribution({a_name: a, b_name: b})
        cfg = self._config(f"two_group::{test}")
        self._show(stats_report.report_for_test(cfg, self._bundle, res,
                                                self._figs), cfg, res.diagnostics)

    def _run_multi_group(self) -> None:
        spec = self._feature_spec()
        by = feature_by_condition(compute_feature(self._bundle, spec), self._bundle)
        names = list(by)
        groups = [by[n] for n in names]
        # parametric ANOVA + non-parametric Kruskal, both shown
        res = stats_tests.oneway_anova(groups, names, alpha=self.alpha.value())
        kw = stats_tests.kruskal(groups, names, alpha=self.alpha.value())
        res.extra["kruskal_p"] = kw.p_value
        res.interpretation += (f" Non-parametric Kruskal-Wallis "
                               f"p={kw.p_value:.3g}.")
        self._plot_distribution(by)
        cfg = self._config("anova_oneway")
        self._show(stats_report.report_for_test(cfg, self._bundle, res,
                                                self._figs), cfg, res.diagnostics)

    def _run_anova(self) -> None:
        spec = self._feature_spec()
        vals = compute_feature(self._bundle, spec)
        atype = self.anova_type.currentText()
        alpha = self.alpha.value()
        correction = self.correction.currentText()
        posthoc = self.posthoc.isChecked()
        extra_diags: list[Diagnostic] = []

        if atype in ("Repeated-measures", "Friedman (non-parametric RM)"):
            subj_key = self.subject_unit.currentData()
            agg, dropped, subj_name = stats_anova.subject_condition_frame(
                self._bundle, vals, subject=subj_key)
            n_subj = int(agg["subject"].nunique()) if len(agg) else 0
            if dropped:
                extra_diags.append(Diagnostic(
                    "warning", f"{dropped} subject(s) lacked all conditions and "
                    f"were dropped (RM/mixed needs a fully-crossed design)."))
            if n_subj < 2:
                extra_diags.append(Diagnostic(
                    "warning", "Repeated-measures needs >=2 subjects/units that "
                    "each appear in every condition. With one unit (e.g. a single "
                    "session), this falls back to a trial-level one-way ANOVA — an "
                    "exploratory approximation, not a true RM-ANOVA."))
                res = stats_anova.one_way(
                    stats_anova.trial_frame(self._bundle, vals), alpha=alpha,
                    posthoc=posthoc, correction=correction)
            else:
                res = stats_anova.repeated_measures(
                    agg, within="condition", subject="subject", alpha=alpha,
                    posthoc=posthoc, correction=correction,
                    parametric=(atype == "Repeated-measures"))
        elif atype == "Factorial (two-way)":
            if self.metavar.count() == 0:
                QtWidgets.QMessageBox.information(
                    self, "No 2nd factor", "This session has no numeric trial "
                    "metadata to use as a second factor (median-split).")
                return
            key = self.metavar.currentText()
            df = stats_anova.trial_frame(self._bundle, vals, factor2=key,
                                         factor2_name=key)
            res = stats_anova.factorial(df, factors=("condition", key), alpha=alpha,
                                        posthoc=posthoc, correction=correction)
        elif atype == "ANCOVA":
            if self.metavar.count() == 0:
                QtWidgets.QMessageBox.information(
                    self, "No covariate", "This session has no numeric trial "
                    "metadata to use as a covariate.")
                return
            key = self.metavar.currentText()
            df = stats_anova.trial_frame(self._bundle, vals, covariate=key,
                                         covariate_name=key)
            res = stats_anova.ancova(df, between="condition", covar=key, alpha=alpha)
        else:  # One-way
            res = stats_anova.one_way(
                stats_anova.trial_frame(self._bundle, vals), alpha=alpha,
                posthoc=posthoc, correction=correction)

        res.diagnostics = list(res.diagnostics) + extra_diags
        self._plot_means(feature_by_condition(vals, self._bundle))
        cfg = self._config(f"anova::{atype}")
        self._show(stats_report.report_for_anova(cfg, self._bundle, res,
                                                 self._figs), cfg, res.diagnostics)

    def _run_correlation(self) -> None:
        if self.metavar.count() == 0:
            QtWidgets.QMessageBox.information(
                self, "No metadata", "This session carries no numeric trial "
                "metadata to correlate with.")
            return
        spec = self._feature_spec()
        x = compute_feature(self._bundle, spec)
        key = self.metavar.currentText()
        y = np.asarray(self._bundle.metadata[key], float)
        res = stats_tests.correlation(x, y, method="pearson", alpha=self.alpha.value())
        self._plot_scatter(x, y, key)
        cfg = self._config(f"correlation::{key}")
        self._show(stats_report.report_for_test(cfg, self._bundle, res,
                                                self._figs), cfg, res.diagnostics)

    def _run_cluster(self) -> None:
        a_name, b_name = self.cond_a.currentText(), self.cond_b.currentText()
        if a_name == b_name:
            QtWidgets.QMessageBox.warning(self, "Pick two conditions",
                                          "Condition A and B must differ.")
            return
        chans = self._channels()
        idx = self._bundle.roi_indices(chans)
        series = self._bundle.X[:, idx, :].mean(axis=1)         # (n_trials, n_times)
        ya = self._bundle.condition_names.index(a_name)
        yb = self._bundle.condition_names.index(b_name)
        A, B = series[self._bundle.y == ya], series[self._bundle.y == yb]
        paired = (self.test.currentText() == "Paired t-test") and A.shape[0] == B.shape[0]
        cluster = mu.cluster_permutation_time(
            A, B, paired=paired, n_permutations=self.nperm.value(),
            seed=self.seed.value(), alpha=self.alpha.value())
        # pointwise map (corrected vs uncorrected) for honesty
        umap = mu.mass_univariate_test(A, B, paired=paired,
                                       correction=self.correction.currentText(),
                                       alpha=self.alpha.value())
        self._plot_cluster(a_name, b_name, A, B, cluster)
        cfg = self._config("cluster_permutation_time")
        diags = list(cluster.diagnostics) + [
            Diagnostic("info",
                       f"Pointwise: {umap.n_significant_uncorrected} timepoints "
                       f"uncorrected vs {umap.n_significant_corrected} after "
                       f"{self.correction.currentText()}.")]
        self._show(stats_report.report_for_cluster(cfg, self._bundle, cluster,
                                                   self._figs), cfg, diags)

    # ----- plots --------------------------------------------------------- #
    def _new_axes(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#14181c")
        for s in ax.spines.values():
            s.set_color("#5a6672")
        ax.tick_params(colors="#cdd4da")
        ax.xaxis.label.set_color("#cdd4da")
        ax.yaxis.label.set_color("#cdd4da")
        ax.title.set_color("#cdd4da")
        return ax

    def _plot_distribution(self, by: dict) -> None:
        self._figs = []
        if self.canvas is None:
            return
        ax = self._new_axes()
        names = list(by)
        rng = np.random.default_rng(0)
        for i, n in enumerate(names):
            v = by[n]
            color = _PALETTE[i % len(_PALETTE)]
            ax.boxplot(v, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.35, color=color),
                       medianprops=dict(color="#ffffff"),
                       whiskerprops=dict(color=color), capprops=dict(color=color),
                       flierprops=dict(markeredgecolor=color))
            ax.scatter(np.full(v.size, i) + rng.uniform(-0.12, 0.12, v.size), v,
                       s=12, color=color, alpha=0.7, edgecolors="none")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("feature")
        ax.set_title("Feature by condition")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._figs = [(self.fig, "Feature distribution by condition")]

    def _plot_means(self, by: dict) -> None:
        self._figs = []
        if self.canvas is None:
            return
        ax = self._new_axes()
        names = list(by)
        means = [float(np.mean(by[n])) for n in names]
        sems = [float(np.std(by[n], ddof=1) / np.sqrt(len(by[n])))
                if len(by[n]) > 1 else 0.0 for n in names]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
        ax.bar(range(len(names)), means, yerr=sems, color=colors, alpha=0.8,
               capsize=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("feature (mean ± SEM)")
        ax.set_title("Condition means")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._figs = [(self.fig, "Condition means ± SEM")]

    def _plot_scatter(self, x, y, ylabel) -> None:
        self._figs = []
        if self.canvas is None:
            return
        ax = self._new_axes()
        ax.scatter(x, y, s=14, color=_PALETTE[0], alpha=0.7, edgecolors="none")
        if np.std(x) > 0:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 50)
            ax.plot(xs, np.polyval(coef, xs), color="#e0823a", lw=2)
        ax.set_xlabel("feature")
        ax.set_ylabel(ylabel)
        ax.set_title("Feature vs " + ylabel)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._figs = [(self.fig, f"Feature vs {ylabel}")]

    def _plot_cluster(self, a_name, b_name, A, B, cluster) -> None:
        self._figs = []
        if self.canvas is None:
            return
        ax = self._new_axes()
        t = self._bundle.times
        for series, name, color in ((A, a_name, _PALETTE[0]),
                                    (B, b_name, _PALETTE[1])):
            m = series.mean(0)
            sem = series.std(0, ddof=1) / np.sqrt(series.shape[0])
            ax.plot(t, m, color=color, lw=2, label=name)
            ax.fill_between(t, m - sem, m + sem, color=color, alpha=0.2)
        mask = cluster.significance_mask()
        if mask.any():
            ymin, ymax = ax.get_ylim()
            ax.fill_between(t, ymin, ymax, where=mask, color="#e0c040", alpha=0.18,
                            label="sig. cluster")
        ax.axvline(0, color="#777777", ls="--", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.set_title(f"{a_name} vs {b_name} — cluster permutation")
        ax.legend(facecolor="#1b2026", labelcolor="#cdd4da", edgecolor="#5a6672")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._figs = [(self.fig, "Condition ERPs with significant time clusters")]

    # ----- show / export ------------------------------------------------- #
    def _show(self, report, config, diagnostics) -> None:
        self._last = (config, report)
        self.summary.setHtml(report.to_html())
        self.diag.setHtml("<br>".join(
            f'<span style="color:{"#b00020" if d.level=="error" else "#9a6b00" if d.level=="warning" else "#3a6ea5"}">'
            f'[{d.level}]</span> {d.message}' for d in diagnostics) or "No warnings.")
        self.tabs.setCurrentIndex(0)

    def _export(self, kind: str) -> None:
        if self._last is None:
            QtWidgets.QMessageBox.information(self, "Nothing to export",
                                             "Run an analysis first.")
            return
        config, report = self._last
        filt = {"html": "HTML (*.html)", "csv": "CSV (*.csv)",
                "json": "JSON (*.json)", "png": "PNG (*.png)"}[kind]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export {kind.upper()}", f"stats_report.{kind}", filt)
        if not path:
            return
        try:
            if kind == "html":
                report.save(path)
            elif kind == "json":
                from neurobci.analysis.shared.export import save_json
                save_json(config.to_dict(), path)
            elif kind == "csv":
                from neurobci.analysis.shared.export import save_table_csv
                save_table_csv({"key": list(config.params),
                                "value": [str(v) for v in config.params.values()]},
                               path)
            elif kind == "png" and self.canvas is not None:
                from neurobci.analysis.shared.export import save_figure
                save_figure(self.fig, path)
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved:\n{path}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def _save_history(self) -> None:
        if self._last is None:
            return
        config, _ = self._last
        self._ctl.analysis_history.add(
            config, summary={"type": config.analysis_type,
                             "conditions": config.selection.get("conditions")})
        if hasattr(self._ctl, "history_ws"):
            self._ctl.history_ws.refresh()
        QtWidgets.QMessageBox.information(self, "Saved", "Added to Analysis History.")

    def update_view(self) -> None:
        pass
