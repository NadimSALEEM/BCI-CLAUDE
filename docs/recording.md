# Session recording format

A recording is a **directory** (not a single file) so that samples can be
streamed to disk continuously and a crash loses at most the last unflushed
chunk. The session remains readable even if it was never closed.

```
<recording_dir>/<participant>_<YYYYmmdd_HHMMSS>/
├── metadata.json     self-describing header (see below)
├── eeg.f32           (n_samples × n_channels) C-order float32, microvolts
├── timestamps.f64    (n_samples,) float64 source/LSL timestamps
└── markers.jsonl     one {"t","label","sample"} JSON object per line
```

## `metadata.json`

Written (partial, `finalised:false`) at `start()` and rewritten
(`finalised:true`, with sample counts and duration) at `stop()`. Contains
everything needed to reproduce the session:

| Field | Meaning |
|-------|---------|
| `format_version`, `software_version` | provenance |
| `created_utc`, `ended_utc` | timestamps |
| `participant_id`, `notes` | **pseudonymous** id + free text (no real names) |
| `source_kind`, `stream_name`, `units` | `simulated`/`lsl`, stream name, `uV` |
| `sfreq_nominal`, `n_channels` | acquisition rate, channel count |
| `channel_names`, `channel_kinds` | montage layout (eeg/eog/misc) |
| `dtype` | `float32` |
| `config` | **full `AppConfig` snapshot** (acquisition, simulation seed, preprocessing, …) |
| `n_samples`, `duration_s`, `first_timestamp`, `last_timestamp` | added at finalisation |

## Reading a session

```python
from neurobci.recording import load_session, to_mne_raw, export_fif, export_npz

s = load_session("recordings/anon_20260617_103000")
print(s.data.shape, s.sfreq, len(s.markers))   # (N, C), 500.0, ...

raw = to_mne_raw(s)        # mne.io.RawArray, EEG/EOG typed, µV→V, markers→annotations
export_fif(s, "out_raw.fif")
export_npz(s, "out.npz")
```

### Crash recovery

If `metadata.json` has `finalised:false` (process died mid-recording),
`load_session` recovers `n_samples` from the size of `eeg.f32`
(`size / (4 × n_channels)`) and trims any partial trailing row. Timestamps
shorter than the EEG file are padded with `NaN`.

## Units & MNE conventions

Samples are stored in **microvolts**. `to_mne_raw` multiplies by `1e-6`
because MNE works in volts, and maps `channel_kinds` to MNE channel types
(`eog` for the EOG channel, `eeg` otherwise). Markers become
`mne.Annotations` with onsets relative to the first sample timestamp,
which makes epoching in later phases straightforward.

## CLI

```bash
python scripts/export_session.py <session_dir> [--fif] [--npz]
```
