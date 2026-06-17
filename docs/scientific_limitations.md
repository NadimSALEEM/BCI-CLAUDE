# Scientific limitations

Read this before trusting any output. NeuroBCI is a **research tool**, not a
medical device, and decodes only pre-defined, calibrated responses.

## General

- **Short calibration ⇒ small datasets.** Models are simple/regularised on
  purpose; deep learning is not a default. Performance is reported **against
  chance** and a calibration is marked *NOT USABLE* unless the best model
  clearly beats chance.
- **Simulation is not reality.** The synthetic generators embed deliberately
  learnable signals; the high accuracies they yield reflect synthetic SNR,
  **not** any real participant. Honesty checks (chance-level on no-signal
  data) are built into the test suite.
- **Dry electrodes** (Enobio) are more artifact-prone, and **no true
  impedance** is available over LSL — none is ever fabricated. Quality is
  inferred from the signal only.
- **Online ≠ offline.** Causal real-time filtering differs from offline
  zero-phase filtering; the two paths are kept strictly separate and the
  online path never uses future samples.
- **Evaluation guards** prevent the usual inflation: the whole pipeline is
  refit inside each CV fold, block/group-aware splitting is supported, and
  imbalance-robust metrics (balanced accuracy, AUC, κ, MCC) are primary.

## Per paradigm

- **P300** — robust, but single-trial accuracy is modest; reliable selection
  comes from aggregating repetitions. Real single-trial ≈ 70–85%.
- **Motor imagery** — subject-dependent; many users need training. CSP can
  overfit small sets (mitigated by regularisation).
- **SSVEP** — CCA is training-light but needs a few seconds of steady-state
  response and good occipital signal; flicker frequencies must be
  distinguishable given the sampling rate.
- **ErrP** — **not every user produces detectable ErrPs.** It is calibrated
  and evaluated separately and used only as an optional supervisory layer.

## Cognitive-state indices

The engagement/workload/drowsiness indices are **exploratory proxies, not
measurements of mental state.** Every index shows its formula, the channels
used and the raw band powers behind it, and is flagged invalid when
contaminated. Generic indices **require experimental validation** for your
task and participant. A supervised state classifier (which would need
labelled data) is intentionally not provided.

## Topography

The 10-20 topomap positions are a standard approximation (not device-measured
or individualised) and are for visualisation only — not source localisation.
