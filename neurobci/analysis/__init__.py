"""Offline analysis suite: statistics and machine learning over EEG.

Two user-facing modules (Statistics, Machine Learning) sit on top of the same
headless cores in this package, which in turn reuse the platform's existing
epoching / ERP machinery (:mod:`neurobci.bci.epoching`,
:mod:`neurobci.bci.erp_analysis`) and preprocessing pipeline. Nothing here
touches live acquisition or the live preprocessing config.

Sub-packages:

* :mod:`neurobci.analysis.shared` -- reproducible config, analysis history,
  optional-dependency detection, data validation/diagnostics, exporters.
* :mod:`neurobci.analysis.stats` -- trial-level feature extraction, parametric
  and non-parametric tests, EEG mass-univariate / cluster-permutation stats.
* :mod:`neurobci.analysis.ml` -- model registry, leakage-safe pipelines and
  model evaluation (cross-validation, chance-level testing, diagnostics).
"""
