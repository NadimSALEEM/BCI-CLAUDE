# Replay & advanced simulation

Phase 8 completes the source abstraction with a third backend and adds
virtual streaming, artifact injection and ground-truth round-trips.

## ReplaySource

`acquisition/replay_source.py` plays a recorded session back as an
`EEGSource`. Because it satisfies the same contract as the simulated and
LSL sources, **every** downstream module (buffer, raw view, quality,
preprocessing, spectral, calibration) works on replayed data unchanged.

Transport:

| Control | Method |
|---------|--------|
| play/pause | `pause()` / `resume()` |
| speed | `set_speed(x)` (0.5–4×, or any factor) |
| seek | `seek(seconds)` |
| restart | `restart()` |
| loop | `loop=True` |

Two timing modes:

- **realtime** (default) — emits samples paced by the wall clock × speed;
- **deterministic** (`realtime=False`) — emits fixed blocks; `read_all()`
  returns the whole session exactly, used by regression tests.

Original markers are re-surfaced via `poll_markers()` (and re-published on a
marker outlet when streamed over LSL).

## Importing external recordings (XDF / FIF)

A `ReplaySource` is built from a `LoadedSession`, so any file we can decode
into that structure becomes a replayable — and re-publishable — source.
`recording/external.py` adds two importers beside the native directory
format:

- **`.xdf` / `.xdfz`** (LabRecorder / LSL) — `load_xdf()` picks the EEG data
  stream (preferring one typed `EEG`, else the widest regular stream),
  recovers channel labels/units from the stream description, rescales to
  microvolts, and maps every string ("Markers") stream onto EEG sample
  indices by timestamp. A missing `nominal_srate` is estimated from the
  timestamps. Channel kinds (eeg/eog/misc) are auto-detected from the
  electrode names (see *Preprocessing → automatic montage detection*), so
  EOG/aux leads are split out even from a blanket-`EEG` file.
- **`.fif` / `.fif.gz`** (MNE-Python / BIDS) — `load_fif()` reads the raw,
  converts volts→microvolts, carries channel types (EEG/EOG/misc) and turns
  annotations into markers. This round-trips `export_fif()`.

`load_session_any(path)` dispatches on what the path is (directory → native,
`.xdf`/`.fif` → the importers), and `ReplaySource`, the engine factory and
the Replay tab all route through it — so an imported file plays back and
streams over LSL identically to a native recording. In the GUI use
**Load file (XDF/FIF)…** next to **Load session…**. Both `pyxdf` and `mne`
are imported lazily and only when that format is opened.

## Engine integration

`source_type="replay"` + `acquisition.replay_path/replay_speed/replay_loop`
in the config make replay a first-class mode. The acquisition thread treats
simulated **and** replayed sources as self-clocked (status `REPLAYED`, no
false "stale" warnings). The **Replay** GUI tab drives all of this:
load a session, play/pause/seek/loop/speed, optionally inject artifacts, and
optionally re-publish as a virtual LSL stream.

## Virtual LSL stream

`acquisition/lsl_publisher.py` re-broadcasts any source onto an LSL outlet
(with proper channel labels/types), so another tool — or a second instance
of this app — can consume it like real hardware. A loopback test
(`tests/test_lsl_publisher.py`) publishes and receives it back to verify the
path end-to-end (skips gracefully if the environment blocks LSL).

## Configurable artifact injection

`acquisition/artifacts_inject.py` overlays line noise, eye-blinks (EOG +
frontal), amplitude bursts, drift and flat channels onto a chunk — used by
replay to stress-test preprocessing/quality on otherwise-clean recordings.

## Ground-truth simulation & regression

`recording/record_p300_session` runs the simulator (injecting a real P300 on
target stimuli) and writes a normal session to disk. The regression test
then does the full loop:

```
simulate → record → load → replay (== recording) → epoch → decode
```

and asserts the planted P300 is recovered above chance — exercising
acquisition, recording, replay and the BCI stack together.

## Programmatic use

```python
from neurobci.acquisition import ReplaySource, LSLPublisher
from neurobci.recording import load_session, record_p300_session

path = record_p300_session("recordings_sim", n_stimuli=200)
rs = ReplaySource(load_session(path), speed=2.0, loop=True)
LSLPublisher(rs, stream_name="NeuroBCI-Replay").start()   # virtual device
```

Re-publish an externally recorded `.xdf` (or `.fif`) over LSL — the path is
all that changes, since `ReplaySource` accepts a directory, an XDF or a FIF:

```python
from neurobci.acquisition import ReplaySource, LSLPublisher

rs = ReplaySource("sub-P025_ses-S001_task-oddball_run-001_eeg.xdf", loop=True)
LSLPublisher(rs, stream_name="NeuroBCI-Replay").start()
```
