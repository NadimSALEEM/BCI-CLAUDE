# Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `python` opens the Microsoft Store | Windows App-Execution alias — call python by full path or disable the alias (see [installation.md](installation.md)). |
| `import pylsl` fails / "liblsl not found" | Native liblsl missing. Only needed for live LSL / virtual-LSL; use Simulation/Replay otherwise. |
| "No matching LSL stream found" | Headset/stream not running, or name/type mismatch. Use **Scan for LSL streams** and pick from the list; leave the name blank to auto-discover by type. |
| Channels tab shows red errors | Stream montage/rate ≠ configured montage. Fix the EEG/EOG labels and confirm the sampling rate; errors block calibration on purpose. |
| Signal Quality shows BAD channels | Flat/railing/noisy electrode. Reseat/clean the electrode; bad channels are excluded from indices and topomaps. |
| Calibration says "NOT USABLE" | The best model didn't beat chance — insufficient/contaminated data. Collect more trials, improve signal quality, or raise the synthetic amplitude in simulation. This is intentional honesty, not a bug. |
| Control demo: nothing happens | Need a **P300** model (Calibration tab) **and** a running stream. If the stream is disconnected or the emergency stop is active, the safety layer blocks commands (shown in history). |
| Topomap is blank | matplotlib unavailable, or too few channels with known positions/finite values in this window. |
| GUI won't open over SSH/headless | Set `QT_QPA_PLATFORM=offscreen` for headless runs (used by the smoke tests); a real display is needed for the interactive app. |
| Replay won't start | Load a valid session directory first (must contain `metadata.json` + `eeg.f32`). |
| Tests fail to delete a temp dir on Windows | Harmless OS file-lock race after close; the session tests use `ignore_errors` cleanup. |

If a workspace errors, the app logs it (see `logs/neurobci.log`) and keeps
running — acquisition is never stopped by a UI error.
