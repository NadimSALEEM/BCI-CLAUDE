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
