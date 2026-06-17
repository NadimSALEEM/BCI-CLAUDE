# User guide

NeuroBCI is a desktop platform for real-time EEG/BCI research. It works with
an Enobio headset over LSL **or** entirely in simulation/replay. See
[installation.md](installation.md) first.

> ⚠️ Research tool, **not a medical device**. It decodes only pre-defined,
> calibrated responses. See [scientific_limitations.md](scientific_limitations.md)
> and [safety.md](safety.md).

## The window at a glance

A persistent **status bar** always shows: operating mode, connection health,
sample rate, signal quality, recording state, current prediction and the
**Emergency Stop**. The toolbar has the emergency-stop button. Ten tabs:

| Tab | Use |
|-----|-----|
| Connection / Acquisition | choose source (Simulated / Live LSL), scan & start |
| Channels | validate montage/rate, correct EEG/EOG labels |
| Raw EEG | stacked traces (EOG in amber) |
| Preprocessing | edit the causal/offline pipeline, before/after, artifacts |
| Signal Quality | per-channel explainable quality |
| Spectral / State | PSD, band-power topomap, cognitive-state indices, baseline |
| Calibration | pick a paradigm, run simulated calibration, compare, save |
| Control / BCI | on-screen P300 selection demo through the safety-gated stack |
| Replay | play back a recorded session; inject artifacts; publish virtual LSL |
| Recording | record EEG + markers to disk |

## Core workflows

### 1. Simulation (no hardware)

Acquisition → *Simulated* → **Start**. Every tab now works on synthetic EEG
(clearly flagged `SIMULATED`). Good for learning the tool and development.

### 2. Live acquisition (Enobio / any EEG LSL)

Acquisition → *Live LSL* → **Scan for LSL streams** → pick one → **Start**.
Go to **Channels** to validate the montage and fix any mislabelled EOG.
Watch **Signal Quality** before trusting the data.

### 3. Calibrate a model

**Calibration** → choose a paradigm (P300 / motor imagery / SSVEP / ErrP) →
**Run simulated calibration**. Read the model-comparison table: results are
shown **against chance**, and the panel turns red ("NOT USABLE") if the best
model doesn't clearly beat chance. **Save best model** when usable.

### 4. Online control demo (P300)

Calibrate a P300 model, start a (simulated) stream, then **Control / BCI** →
pick an intended item → **Run selection**. The cursor moves to the selected
item. **Test mode** predicts without acting; the **Emergency Stop** (toolbar)
blocks all command execution. History + online metrics update live.

### 5. Spectral & cognitive-state monitoring

**Spectral / State** shows the PSD, a band-power **topomap** (pick band +
map mode), a temporal index trace and a transparent index panel. **Capture
baseline** to view change-from-baseline. The indices are **exploratory
proxies** — their formula, channels and raw values are always shown.

### 6. Record & replay

**Recording** → set a pseudonymous participant id → **Start recording** →
drop markers → **Stop**. Later, **Replay** → **Load session** → **Start
replay** (play/pause/seek/speed/loop); all other tabs operate on the
replayed data. Optionally inject artifacts or **publish as a virtual LSL
stream**.

## Headless / scripting

```bash
python scripts/calibrate.py 300 --save                 # calibrate from CLI
python scripts/export_session.py recordings/<dir> --fif --npz
python examples/quickstart.py                          # calibrate -> online
python examples/offline_analysis.py                    # record -> analyse -> decode
```

See [docs/](.) for module-specific guides (preprocessing, paradigm_p300,
paradigms_extra, control, spectral, recording, replay, architecture) and
[troubleshooting.md](troubleshooting.md).
