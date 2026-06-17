# Architecture

NeuroBCI is layered and event-driven. The central design rule is:

> **Acquisition must never be blocked by anything else** — not a slow plot,
> not model training, not the UI thread.

This is achieved with a single-writer **ring buffer** between the acquisition
thread and every consumer.

## Real-time data flow (Phase 1)

```
 ┌─────────────────────────────────────────────────────────────┐
 │                       UI thread (Qt)                          │
 │                                                               │
 │   QTimer (refresh_hz, e.g. 30 Hz)                             │
 │        │ pull latest_seconds(W)                               │
 │        ▼                                                      │
 │   Active workspace  ── Raw EEG │ Signal Quality │ Acquisition │
 │   StatusBar (mode, connection, rate, quality, e-stop)         │
 └────────▲──────────────────────────────────────────────────────┘
          │ copy of recent window (non-blocking)
 ┌────────┴──────────────────────────────────────────────────────┐
 │                     RingBuffer  (thread-safe)                  │
 │            capacity = buffer_seconds × sfreq samples           │
 └────────▲──────────────────────────────────────────────────────┘
          │ append(chunk, timestamps)        (single writer)
 ┌────────┴──────────────────────────────────────────────────────┐
 │                  AcquisitionThread (daemon)                    │
 │   loop: source.read() → buffer.append() → update AppState      │
 │         measure effective sfreq, detect staleness/drops        │
 └────────▲──────────────────────────────────────────────────────┘
          │ read() → (samples, timestamps)
 ┌────────┴──────────────────────────────────────────────────────┐
 │   EEGSource (one contract)                                     │
 │     ├── SimulatedSource   (synthetic, ground-truth)            │
 │     ├── LSLSource         (Enobio / any EEG LSL stream)        │
 │     └── ReplaySource      (Phase 8)                            │
 └────────────────────────────────────────────────────────────────┘
```

`AppState` (lock-protected) and a small synchronous `EventBus` carry discrete
status; high-rate sample data goes only through the ring buffer.

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `neurobci.config` | Typed dataclass schema; JSON profile load/save/duplicate. No code execution. |
| `neurobci.core.ring_buffer` | Thread-safe fixed-capacity circular buffer; the acquisition/UI decoupler. |
| `neurobci.core.stream_info` | Source-agnostic stream metadata; EEG vs EOG channel kinds. |
| `neurobci.core.app_state` | Observable mode/connection/recording/prediction/e-stop state. |
| `neurobci.core.events` | Synchronous pub/sub for discrete events (headless-friendly). |
| `neurobci.acquisition.base` | `EEGSource` ABC (`start/read/stop/info`). |
| `neurobci.acquisition.simulated` | Realistic synthetic EEG; deterministic `generate()` + real-time `read()`. |
| `neurobci.acquisition.lsl_source` | LSL backend; lazy `pylsl`; metadata parsing + montage fallback. |
| `neurobci.acquisition.acquisition_thread` | Resilient pull loop; rate/health measurement. |
| `neurobci.acquisition.engine` | Facade wiring source+buffer+state+thread from config. |
| `neurobci.quality.metrics` | Explainable per-channel quality (std, line, HF, railing, flat). |
| `neurobci.ui` | PyQt5 main window, workspaces, status bar, theme. |

## Threading model

- **One** acquisition daemon thread (single writer to the ring buffer).
- The Qt UI thread reads via copies — it never holds the buffer lock beyond a
  numpy copy.
- Exceptions in `source.read()` are caught, logged, and surfaced as a
  `LOST`/`STALE` connection; acquisition keeps retrying rather than dying.

## Why these choices

- **Ring buffer + pull model** keeps the design simple (no backpressure
  plumbing) and guarantees UI jitter cannot corrupt timing.
- **`generate()` vs `read()` split** in the simulator separates deterministic
  *content* (testable, known ground truth) from real-time *pacing*.
- **Lazy `pylsl` import** means simulation users need no native library.
- **stdlib JSON + dataclasses** for config avoids a YAML dependency and any
  risk of executing config content.
