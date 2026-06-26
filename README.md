# NeuroBCI — Real-time EEG / BCI Research Platform

A modular desktop platform for real-time EEG acquisition, signal-quality
monitoring, preprocessing, calibration, classification, decision-making and
safe computer control. It targets an **Enobio** dry-electrode headset over
**Lab Streaming Layer (LSL)** and ships a full **simulation mode** so the
entire application can be developed and used without hardware.

> ⚠️ **Not a medical device.** NeuroBCI decodes only pre-defined,
> experimentally calibrated EEG responses or mental tasks. It does not read
> arbitrary thoughts and must not be used for clinical decisions.

---

## Status — all 9 phases complete

This repository implements the full roadmap **Phase 1 (Core)** … **Phase 9
(validation & packaging)** end to end:

| Area | Status |
|------|--------|
| Project structure, packaging, logging | ✅ |
| Configuration system (typed schema + JSON profiles) | ✅ |
| Application state + event bus | ✅ |
| Thread-safe ring buffer (acquisition/UI decoupling) | ✅ |
| Source abstraction: **Simulated**, **LSL**, **Replay** | ✅ |
| Realistic EEG simulator (alpha/theta/beta, 1/f, line, drift, blinks, faults) | ✅ |
| LSL stream discovery (scan & pick) | ✅ |
| Channel/sfreq validation + manual EEG/EOG correction | ✅ |
| Live signal-quality metrics (explainable, no fake impedance) | ✅ |
| Crash-safe session recording (raw + timestamps + markers + metadata) | ✅ |
| Session loader + MNE `.fif` / `.npz` export | ✅ |
| **Modular preprocessing pipeline (filters, notch, band-stop, CAR, robust/Laplacian re-ref, clamp, smoothing, z-score, detrend, Sav–Golay)** | ✅ |
| **Strict causal-online vs zero-phase-offline separation** | ✅ |
| **Add/remove/reorder/enable/edit stages + validation warnings + profiles** | ✅ |
| **One shared live pipeline → preprocessed stream consumed by quality/spectral/raw tabs** | ✅ |
| Artifact detection (flat/clip/pop/EMG/line/blink/missing) with reasons | ✅ |
| **Calibrated artifact removal: bad-channel interpolation, ICA, ASR (fit-on-calibration, applied online; pass-through until fitted)** | ✅ |
| **Data-trust verification (TRUST/CAUTION/UNTRUSTWORTHY) for windows & recordings + `verify_recording.py` CLI** | ✅ |
| Before/after live visualization | ✅ |
| **P300 paradigm: epoching, xDAWN/LDA + Riemannian models** | ✅ |
| **Leakage-free CV (balanced-acc/AUC/κ/MCC) reported vs chance** | ✅ |
| **Calibration → model comparison → save; honest "not usable" gating** | ✅ |
| Online P300 decoding (per-item selection aggregation) | ✅ |
| Model persistence + compatibility checks; simulator ERP injection | ✅ |
| **Decision layer: selection early-stop + streaming vote/refractory/no-command** | ✅ |
| **Safety gating: e-stop / connection / quality / confidence / rate / external-off** | ✅ |
| **Command router + history + online metrics (acc, cmd/min, abstention, latency)** | ✅ |
| On-screen control demo (ring board, cursor) + test mode | ✅ |
| **Motor imagery (active): CSP/Riemannian + streaming decision layer** | ✅ |
| **SSVEP (reactive): training-light CCA decoder** | ✅ |
| **ErrP error-correction layer + flexible command mapping** | ✅ |
| Paradigm-selectable simulated calibration in the GUI | ✅ |
| **Spectral: PSD, abs/relative band power, IAF, regional aggregation** | ✅ |
| **Cognitive-state indices (engagement/workload/drowsiness) — formula + raw values + caveats** | ✅ |
| Band-power topomaps + baseline-relative change + temporal traces | ✅ |
| **Replay recorded sessions (play/pause/seek/speed/loop) — all tabs work on it** | ✅ |
| **Virtual LSL re-publishing + configurable artifact injection** | ✅ |
| Ground-truth simulate→record→replay→decode regression | ✅ |
| **Performance profiling (preprocessing ~750× real-time, inference <0.3 ms)** | ✅ |
| **Packaging (`python -m neurobci`, console script, LICENSE), example configs + workflows** | ✅ |
| **Consolidated runner + full user/developer/safety docs** | ✅ |
| PyQt5 GUI: 10 workspaces | ✅ |
| Emergency-stop safety flag (surfaced everywhere, gates every command) | ✅ |
| Tests (148 unit/integration) + smoke scripts + profiler | ✅ |

---

## Requirements

- Python ≥ 3.10 (validated on **3.14**)
- The packages in [`requirements.txt`](requirements.txt): numpy, scipy,
  scikit-learn, mne, pyriemann, **pylsl** (needs native `liblsl`), **PyQt5**,
  pyqtgraph, matplotlib.

Install:

```bash
python -m pip install -r requirements.txt
```

`pylsl` requires the native **liblsl** library at runtime; it is only needed
for *live* acquisition. Simulation mode works without it.

> On Windows, if `python` opens the Microsoft Store, call the interpreter by
> its full path or disable the App-Execution alias.

---

## Running

```bash
python run.py                # loads the 'default' profile, starts in simulation
python run.py my_profile     # loads/creates configs/my_profile.json
```

In the **Connection / Acquisition** tab choose *Simulated* or *Live LSL*
(use **Scan for LSL streams** to discover and pick one), then **Start**.
Then:

- **Channels** — customise the montage from the start: even with no stream
  connected, set the channel count, fill names from a preset, rename channels
  to real 10-20 positions, set each kind (EEG/EOG/misc) and **delete channels
  you don't want**; Apply stores it for the next acquisition. With a stream
  connected it relabels/reclassifies and **drops** channels on the *live*
  stream — the deletion propagates everywhere (preprocessing, quality, spectral,
  topomaps, epoching, recording) so analyses never include removed channels.
  It also validates the montage/rate against the configured expectation.
- **Raw EEG** — stacked traces (EOG highlighted in amber).
- **Preprocessing** — edit the pipeline (enable/reorder/edit params),
  switch causal/offline, see the before/after comparison and detected
  artifacts (see [`docs/preprocessing.md`](docs/preprocessing.md)).
- **Signal Quality** — per-channel explainable dashboard.
- **Spectral / State** — PSD, band-power topomap, temporal index trace and a
  transparent cognitive-state index panel (formula + raw values + caveats),
  with baseline capture (see [`docs/spectral.md`](docs/spectral.md)).
- **Calibration** — pick a paradigm (P300 / motor imagery / SSVEP / ErrP),
  run a simulated calibration, compare models against chance, save the best,
  and (P300) verify online selection (see
  [`docs/paradigm_p300.md`](docs/paradigm_p300.md) and
  [`docs/paradigms_extra.md`](docs/paradigms_extra.md)).
- **Control / BCI** — drive an on-screen selection demo through the real
  decision + safety + command stack; test mode and emergency stop gate
  execution (see [`docs/control.md`](docs/control.md)).
- **Replay** — load a recorded session and play it back (play/pause/seek/
  speed/loop) through the engine so every tab works on it; inject artifacts
  or re-publish as a virtual LSL stream (see [`docs/replay.md`](docs/replay.md)).
- **ERP / Epoch Average** — offline analysis of the **Replay tab's** session:
  read its event markers, optionally **combine** markers into new derived
  events, group them into conditions, and compute trial averages (per-channel
  ERP + SEM, butterfly, GFP, electrode-adaptive latency topomap, `.npz`
  export). Reuses the configured preprocessing unchanged or epochs the raw
  signal (see [`docs/analysis.md`](docs/analysis.md)).
- **Recording** — set a pseudonymous participant id + notes, **Start
  recording**, drop event markers, then **Stop**. Sessions are written to
  `recordings/` (see [`docs/recording.md`](docs/recording.md)).

### Exporting a recording

```bash
python scripts/export_session.py recordings/<session_dir> --fif --npz
```

### Calibrating a model (headless)

```bash
python scripts/calibrate.py 300 --save     # simulated P300, saves best to ./models
```

### Headless checks (no display, no hardware)

```bash
python scripts/engine_smoke.py 3                       # sim → buffer → quality
QT_QPA_PLATFORM=offscreen python scripts/gui_smoke.py  # full GUI offscreen
```

### Tests

```bash
python -m unittest discover -s tests -v   # stdlib runner (no extra deps)
# or, if installed:  pytest
```

---

## Operating modes

- **Simulation** — synthetic EEG with known ground truth; clearly flagged.
- **Live (LSL)** — receive from the Enobio (or any EEG LSL stream); channel
  names/types read from stream metadata, with manual-montage fallback.
- **Replay / Offline / Online BCI** — defined in the architecture; implemented
  in later phases.

---

## Architecture (one-paragraph version)

A single **acquisition thread** is the only writer to a **thread-safe ring
buffer**. Sources (`Simulated`, `LSL`, …) all implement one `EEGSource`
contract, so everything downstream is backend-agnostic. The GUI never blocks
acquisition: a single **refresh timer** *pulls* recent windows from the buffer
at the render rate, completely decoupled from the ~500 Hz sample rate. See
[`docs/architecture.md`](docs/architecture.md) for diagrams and the full
module map.

```
neurobci/
  config/        typed schema + JSON profile manager
  core/          logging, ring buffer, stream info, app state, event bus
  acquisition/   EEGSource ABC, simulated/LSL/replay sources, discovery, validation, LSL publisher, artifact injection, engine
  preprocessing/ ordered pipeline (filters, notch, CAR, detrend) + artifact detection
  paradigms/     framework + P300 / motor-imagery / SSVEP / ErrP + synthetic generators
  bci/           epoching, features, models, CCA, evaluation, calibration, online, model store
  control/       decision layer, safety, adapters, router, metrics, mapping, ErrP correction
  spectral/      PSD, band power, cognitive-state indices, topomap, analyzer
  quality/       explainable signal-quality metrics
  recording/     crash-safe session writer + loader/exporter (MNE/npz)
  ui/            PyQt5 main window, workspaces, widgets, theme
tests/           unit + integration tests (unittest)
scripts/         headless engine/GUI smoke, session export, calibration CLI
configs/         saved configuration profiles (JSON)
recordings/      recorded sessions (created at runtime)
models/          saved calibrated models (created at runtime)
```

---

## Scientific limitations (read before trusting results)

- **Short calibration ⇒ small datasets.** Later phases must report
  chance-level honestly and never present it as success.
- **Dry electrodes** are more artifact-prone; **no true impedance** is
  available over LSL and none is fabricated.
- **Cognitive-state indices** (engagement/fatigue/workload) are *exploratory
  proxies*, not directly observable states; they require experimental
  validation and are always shown with their definitions and raw band powers.
- **Causal real-time** filtering ≠ offline zero-phase filtering; the two are
  kept strictly separate to avoid leakage. The online path never uses future
  samples.

## Safety

External computer/game control (later phases) is **disabled by default**,
rate-limited, requires explicit activation, and halts on stream loss,
low quality, low confidence or instability. The **Emergency Stop** latch is
present from Phase 1 and surfaced in the status bar at all times.

---

## Roadmap

1. **Core application** ✅
2. **Acquisition** ✅ — LSL discovery, channel validation, crash-safe recording, markers, MNE export
3. **Preprocessing** ✅ — modular causal/offline pipeline, artifact detection, before/after, profiles
4. **P300 / ERP selection** ✅ — calibration, xDAWN/LDA + Riemannian, leakage-free CV vs chance, model save, online selection
5. **Control demo** ✅ — decision layer, safety gating, command router + history + online metrics, on-screen selection task
6. **Additional paradigms** ✅ — motor imagery (active), SSVEP (CCA), ErrP correction, flexible command mapping
7. **Spectral & cognitive-state monitoring** ✅ — PSD/band power, IAF, transparent indices, topomaps, baselines
8. **Replay + advanced simulation** ✅ — session replay (transport), virtual LSL, artifact injection, ground-truth regression
9. **Validation & packaging** ✅ — consolidated runner, profiling, packaging, example configs/workflows, full docs

## Documentation

- [Installation](docs/installation.md) · [User guide](docs/user_guide.md) · [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md) · [Developer / extension guide](docs/developer.md)
- Modules: [preprocessing](docs/preprocessing.md) · [P300](docs/paradigm_p300.md) · [other paradigms](docs/paradigms_extra.md) · [control & safety logic](docs/control.md) · [spectral/state](docs/spectral.md) · [ERP/epoch analysis](docs/analysis.md) · [recording](docs/recording.md) · [replay](docs/replay.md)
- **Read before trusting results:** [Scientific limitations](docs/scientific_limitations.md) · [Safety](docs/safety.md)
- Examples: [`examples/quickstart.py`](examples/quickstart.py), [`examples/offline_analysis.py`](examples/offline_analysis.py), [`examples/configs/`](examples/configs/)

## License

MIT (see `pyproject.toml`).
