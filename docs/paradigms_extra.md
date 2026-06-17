# Additional paradigms (Phase 6)

Three more paradigms plug into the same `paradigms` + `bci` + `control`
framework as P300, plus an error-correction layer and flexible command
mapping.

## Motor imagery (active BCI)

`paradigms/motor_imagery.py` — imagine left vs right hand; decoded from the
contralateral sensorimotor **mu/beta power decrease (ERD)**, not an ERP.

- Models: **`csp_lda`** (MNE Common Spatial Patterns + LDA) and
  **`cov_ts_lr`** (covariances → tangent space → logistic regression).
- It is **active/continuous**: per-window class predictions are stabilised
  into commands by the streaming `DecisionLayer` (majority vote + refractory
  + no-command default). `control.simulate_mi_stream` drives this loop.
- Validated: `csp_lda` ≈ 0.85, `cov_ts_lr` ≈ 0.99 balanced accuracy on
  synthetic ERD; ≈ chance with no ERD (honesty check).

## SSVEP (reactive, frequency-tagged)

`paradigms/ssvep.py` — each target flickers at a distinct frequency;
attending one entrains an occipital oscillation at that frequency.

- Decoder: **`bci/cca.py` CCADecoder** — Canonical Correlation Analysis vs
  sine/cosine references (fundamental + harmonics). **Training-light**: no
  per-class classifier fit, so calibration is just a decode-accuracy check.
- Validated: 4-class CCA ≈ 1.0 accuracy on synthetic (chance 0.25).

## ErrP (error-related correction)

`paradigms/errp.py` — after an action, a fronto-central ERN/Pe complex
indicates the user perceived an error.

- Reuses the ERP models (`vec_lda`, `riemann_lr`), calibrated/evaluated
  **separately** from the primary BCI.
- `control/errp_correction.py` **`ErrPCorrector`** classifies the
  post-action epoch and recommends **accept** or **reject** (undo/repeat).
  A supervisory layer, not every user produces detectable ErrPs.
- Validated: detects errors / accepts correct trials on synthetic data.

## Flexible command mapping

`control/mapping.py` **`CommandMap`** decouples *what was decoded* (item /
class index) from *what action it triggers* (`identity`, `directional`, or a
custom dict). The same paradigm can drive a speller, cursor, menu or robot
by swapping the map — no decoding/decision changes.

## Calibration dispatch

`run_simulated_calibration(paradigm, …)` picks the right synthetic
generator and model set per paradigm; SSVEP routes to the CCA decode path.
The **Calibration** GUI tab exposes a paradigm selector that drives this.

## Limitations

- Synthetic generators embed deliberately learnable signals; reported
  numbers reflect synthetic SNR, **not** real-subject performance (real MI
  ≈ 70–85%, real SSVEP varies, real ErrP is subject-dependent).
- The on-screen **Control** demo currently supports **P300 selection**; the
  MI continuous-control and SSVEP/ErrP UIs reuse the same tested stack but
  are not yet surfaced as dedicated on-screen tasks.
