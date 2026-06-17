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
| `bandstop` | Butterworth band-stop / reject (wider than a notch) | `low_hz`, `high_hz`, `order` | ✅ |
| `notch` | IIR notch (mains 50/60 Hz) | `freq_hz`, `quality` | ✅ |
| `moving_average` | FIR moving-average smoother (linear phase) | `window_ms` | ✅ |
| `car` | Common Average Reference (EEG only; EOG excluded) | — | ✅ |
| `robust_ref` | Robust **median** reference (resistant to bad channels) | — | ✅ |
| `laplacian` | Surface Laplacian (Hjorth) — needs 10-20 positions | `neighbors` | ✅ |
| `clamp` | Hard amplitude limiter (±µV) for spike transients | `limit_uv` | ✅ |
| `standardize` | Per-channel z-score (causal EWMA / offline window) | `halflife_s` | ✅ |
| `detrend` | Linear detrend | — | ❌ offline-only |
| `savgol` | Savitzky–Golay polynomial smoothing (centred) | `window_ms`, `polyorder` | ❌ offline-only |

Stages marked **not real-time safe** are automatically **skipped in causal
mode** and reported in the UI, so you always know exactly what ran.

Notes:

- `laplacian` re-references each scalp channel against its nearest neighbours
  using the 10-20 layout; channels with no known position (EOG, or unknown
  names) pass through unchanged, and it self-reports as inactive if no
  positions are available.
- `standardize` outputs **standard-deviation units, not microvolts** — its
  causal path uses an exponential moving mean/variance so it never peeks ahead.

Add, remove, reorder and edit stages live in the **Preprocessing** workspace
(`Add…` lets you insert any of the types above).

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
workspace (enable/disable, ↑/↓ reorder, *Add…* / *Remove*, *Edit…* for
documented parameters), switch causal/offline, and watch the **before /
after** comparison live.

## One shared, live pipeline (used by every tab)

The Preprocessing workspace is **not** a sandbox. The engine owns a single
pipeline, runs it on the acquisition thread (causal, stateful — so streaming
is exact), and stores the output in a **second ring buffer** alongside the raw
one. Every consumer can therefore read cleaned data:

```python
raw, ts  = engine.latest_seconds(2.0)            # raw ring buffer
proc, ts = engine.latest_processed_seconds(2.0)  # after the causal pipeline
```

- **Signal Quality** and **Spectral / State** default to the **preprocessed**
  stream (with a `Source:` toggle back to raw). **Raw EEG** defaults to raw
  with an opt-in *Show: Preprocessed*.
- Editing the pipeline (add/remove/reorder/param/enable) takes effect on the
  live stream **immediately** — edits are applied under a lock the acquisition
  thread shares, and filter state is reset so a removed stage cannot leak.
- In causal mode the workspace's **"after"** plot *is* this shared stream.
- **Safety/electrode quality** (the status-bar rating that gates command
  execution) stays on **raw** on purpose: a notch/CAR/clamp stage could
  otherwise mask a genuine electrode fault.
- The simulated **Calibration**/**Control** demos generate their own
  ground-truth epochs internally, so they are trained and tested in the same
  domain and do not read this live buffer.

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
something was rejected.

## Calibrated artifact removal (bad channels, ICA, ASR)

Three stages *learn* from data and then clean it. They live in
[`artifact_removal.py`](../neurobci/preprocessing/artifact_removal.py) and
follow one rule: **they pass the signal through unchanged until calibrated**,
so an un-fitted remover can never silently distort the live stream. Fitting
happens off the real-time path (on an explicit calibration window) and yields
a fixed linear transform applied causally online — *not* per-window refitting.

| `type` | What it does | Calibrate on |
|--------|--------------|--------------|
| `interpolate_bad` | Detects railing/flat/wild EEG channels and replaces them by neighbour interpolation. **Put it before `car`** so one 2000 µV electrode can't poison the average. | any window |
| `ica` | FastICA; flags components by EOG correlation + excess kurtosis, removes them as a fixed projection. | ~10 s, ideally with some blinks |
| `asr` | Calibrated subspace reconstruction (Mullen 2015-style, *simplified*): repairs high-variance transients (movement/pops) from the clean subspace. | a **clean** stretch (resting) |

Calibrate them from the **Preprocessing** tab → *Calibrate artifact removal*,
or headlessly:

```python
summaries = engine.calibrate_artifacts(seconds=10.0)   # fits all fit-stages
# or, on a recording:  pipeline.fit(window)
```

Honesty: these are research-grade helpers, not magic. The ICA component picker
and the simplified ASR will not match EEGLAB/`asrpy` exactly — **always
verify** the result (below). A stage that could not be fitted, or that found
nothing to remove, says so and passes data through.

> **A single bad electrode** (the classic Enobio failure: one channel railing
> at ±2000 µV) is best handled by ordering the pipeline as
> `highpass → interpolate_bad → notch → (ica) → car`, or by swapping `car` for
> `robust_ref` (median reference), which a wild channel cannot drag around.

## Is my data trustworthy? (`verify`)

[`neurobci.quality.verify`](../neurobci/quality/verify.py) gives a blunt,
explained verdict — **TRUST / CAUTION / UNTRUSTWORTHY** — for a window or a
whole recording. It checks non-finite samples, saturation, unusable-channel
fraction, amplitude plausibility and whether the spectrum falls off like EEG
(1/f) rather than broadband/EMG. Every downgrade names the metric responsible.

```bash
python scripts/verify_recording.py <session_dir>               # raw verdict
python scripts/verify_recording.py <session_dir> --preprocess  # + after pipeline
```

Use this on your real Enobio recordings before believing any index or decode.

## Programmatic use

```python
from neurobci.preprocessing import Pipeline
from neurobci.config.schema import PreprocessingConfig

pipe = Pipeline.from_config(PreprocessingConfig(), sfreq=500.0, ch_kinds=kinds)
online = pipe.process_chunk(chunk)            # causal, stateful
offline = pipe.apply_window(window, mode="offline")   # zero-phase
print(pipe.validate())                        # warnings
```
