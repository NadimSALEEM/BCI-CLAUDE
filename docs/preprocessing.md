# Preprocessing

Preprocessing is an **ordered, configurable pipeline** of stages. The same
configuration drives two strictly separated execution paths:

| Path | Method | Uses future samples? | Use case |
|------|--------|----------------------|----------|
| **Causal / online** | `process_chunk` (stateful), `apply` (window) | **No** | live system, real-time decisions |
| **Offline** | `apply_window(mode="offline")` | Yes (whole window) | analysis of recorded data |

This separation is deliberate: zero-phase (`filtfilt`) filtering and linear
detrending look great offline but **cannot** be used online without peeking
into the future, which would invalidate any real-time BCI result.

## Stages

Configured as a JSON list (`config.preprocessing.stages`), each entry
`{type, enabled, params}`:

| `type` | Description | Key params | Real-time safe |
|--------|-------------|-----------|:--:|
| `highpass` | Butterworth high-pass (drift/DC removal) | `cutoff_hz`, `order` | ✅ |
| `lowpass` | Butterworth low-pass (EMG/anti-alias) | `cutoff_hz`, `order` | ✅ |
| `bandpass` | Butterworth band-pass | `low_hz`, `high_hz`, `order` | ✅ |
| `notch` | IIR notch (mains 50/60 Hz) | `freq_hz`, `quality` | ✅ |
| `car` | Common Average Reference (EEG only; EOG excluded) | — | ✅ |
| `detrend` | Linear detrend | — | ❌ offline-only |

Stages marked **not real-time safe** are automatically **skipped in causal
mode** and reported in the UI, so you always know exactly what ran.

### Causal streaming is exact

IIR filters carry their state (`zi`) across chunks. Feeding a signal in
arbitrary pieces through `process_chunk` yields a result **identical**
(to numerical precision) to filtering the whole signal at once — verified
in `tests/test_preprocessing_filters.py`.

## Default pipeline

```
highpass 0.5 Hz  →  notch 50 Hz  →  lowpass 40 Hz  →  CAR
```

A good general-purpose causal chain. Edit/reorder it in the **Preprocessing**
workspace (enable/disable, ↑/↓ reorder, *Edit…* for documented parameters),
switch causal/offline, and watch the **before / after** comparison live.

## Validation

`Pipeline.validate()` surfaces questionable settings, e.g.:

- a cutoff at/above Nyquist,
- a band-pass with `low ≥ high`,
- a **high-pass cutoff ≥ low-pass cutoff** (empty pass-band — a classic
  mistake that silently deletes the signal).

## Artifact detection

`detect_artifacts(window, stream_info)` returns an **`ArtifactReport`** of
events, each with a channel, a numeric value and a readable reason:

- per-channel: `flat`, `clipping`, `gradient` (electrode pop), `amplitude`,
  `muscle` (HF/EMG), `line_noise`;
- global: `blinks` (counted on EOG), `missing_samples` (non-finite).

Detection never alters the signal — it only *flags*, so you can inspect why
something was rejected. Component-based removal (ICA / ASR) is intentionally
deferred to a later phase and will use **fitted** transforms (fit on
calibration data, apply online), not per-window refitting.

## Programmatic use

```python
from neurobci.preprocessing import Pipeline
from neurobci.config.schema import PreprocessingConfig

pipe = Pipeline.from_config(PreprocessingConfig(), sfreq=500.0, ch_kinds=kinds)
online = pipe.process_chunk(chunk)            # causal, stateful
offline = pipe.apply_window(window, mode="offline")   # zero-phase
print(pipe.validate())                        # warnings
```
