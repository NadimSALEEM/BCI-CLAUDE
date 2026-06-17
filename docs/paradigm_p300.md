# P300 paradigm & the BCI pipeline

Phase 4 implements the first complete BCI paradigm: **P300 / ERP-based
selection**, plus the paradigm-agnostic machinery (epoching, features,
models, evaluation, persistence, online decoding) that future paradigms
reuse.

## Why P300 first

The P300 is a large, stereotyped centro-parietal positive deflection
~300 ms after an attended (rare/target) stimulus. It needs little user
training and is robust, so it exercises the **entire** pipeline end-to-end
with the most reliable signal — the ideal reference implementation.

## Data flow

```
stimuli + markers ─► epoching ─► (features / spatial filter) ─► classifier
                                                                    │
                       per-flash target scores ◄────────────────────┘
                                   │
                 aggregate per item over repetitions ─► selected item
```

## Components

| Module | Responsibility |
|--------|----------------|
| `paradigms/base.py` | `Paradigm` ABC, `EpochWindow`, registry |
| `paradigms/p300.py` | P300 classes, epoch window (−0.1…0.6 s, baseline −0.1…0), selection aggregation |
| `paradigms/synthetic.py` | ground-truth P300 epochs / selection runs (testing + simulated calibration) |
| `bci/epoching.py` | epoch extraction, baseline correction, peak-to-peak rejection |
| `bci/features.py` | `EpochVectorizer` (decimate + flatten) |
| `bci/models.py` | model registry + `ParadigmModel` wrapper |
| `bci/evaluation.py` | leakage-free CV, metrics vs chance, model comparison |
| `bci/model_store.py` | pickle + metadata, compatibility checks |
| `bci/calibration.py` | calibrate = epochs → compare → refit best → report |
| `bci/online.py` | `OnlineP300Decoder`, per-item score accumulation |

## Models

* **`vec_lda`** — decimate+flatten → standardise → shrinkage LDA. Simple,
  fast, interpretable; a strong ERP default.
* **`riemann_lr`** — xDAWN covariances → tangent space → logistic
  regression (pyriemann). Robust Riemannian classifier.

Deep learning is intentionally **not** a default: short calibration yields
only a few hundred epochs, where it would overfit.

## Scientifically-valid evaluation

`bci/evaluation.py` guarantees:

- the **whole** pipeline (spatial filters, scalers, classifier) is refit
  **inside each CV fold** — nothing touches held-out data;
- **block/group-aware** splitting is supported (no temporal leakage);
- every result is reported **against chance** (0.5 balanced accuracy and the
  majority-class accuracy) with class counts;
- imbalance-robust metrics are primary: **balanced accuracy, ROC-AUC,
  Cohen's κ, MCC**, plus the confusion matrix. Plain accuracy is shown but
  never used alone.

`CalibrationResult.is_usable` is **False** unless the best model beats
chance by more than one fold-to-fold standard deviation — chance-level
calibration can never be mistaken for success. (Verified by a test that
trains on pure noise and asserts "not usable".)

## Usage

GUI: **Calibration** tab → choose models → *Run simulated calibration* →
inspect the comparison table + confusion vs chance → *Save best model* →
*Online verification (sim)*.

CLI / API:

```bash
python scripts/calibrate.py 300 --save
```
```python
from neurobci.bci import run_simulated_calibration, OnlineP300Decoder
from neurobci.paradigms.base import get_paradigm
from neurobci.config.schema import ChannelConfig

res = run_simulated_calibration(get_paradigm("p300"), ChannelConfig(), n_trials=300)
print(res.results[res.best_model_name].summary(), res.is_usable)

decoder = OnlineP300Decoder(res.model)
selected, per_item = decoder.decide_selection(X, item_ids, n_items=6)
```

## Live calibration

The simulator can inject a real P300 on cue
(`SimulatedSource.inject_erp(amp)`, centro-parietal, ~300 ms), so a live
cued-stimulus calibration that records markers and epochs from the buffer
is possible. The on-screen stimulus presentation + live epoching UI is
wired up in **Phase 5** (control demo); the decoding stack it needs is
complete and tested now.

## Limitations

- Synthetic P300 is deliberately learnable; reported numbers reflect the
  synthetic SNR, **not** a claim about any real participant. Real
  single-trial P300 is typically ~70–85 % balanced accuracy.
- ICA/ASR artifact *removal* is still deferred (artifacts are detected and
  bad epochs rejected, which is what matters most for ERP).
