# ERP / Epoch-Average analysis

The **ERP / Epoch Average** tab is an offline analysis workspace. It analyses
the session **currently loaded in the Replay tab**, reads the event markers it
carries, lets you group/combine those markers into conditions, and computes
trial-averaged evoked responses — for inspecting P300s, comparing conditions,
drawing scalp maps, and exporting grand averages for downstream analysis.

It is **read-only**: it never touches acquisition and never changes the live
preprocessing configuration.

> **Source = the Replay tab.** This tab no longer opens files itself. Load a
> recording (native dir / `.xdf` / `.fif`) in the **Replay** tab first, then
> here click **Load from Replay tab**. Keeping a single import path means the
> analysis always matches what you're replaying.

> **Montage is inherited from the Channels tab.** On load, the file's channels
> are relabelled to the montage you curated (live Replay stream first, else the
> configured montage). If you **deleted** channels in the Channels tab while
> replaying, the exact same columns are dropped here too — so the channel
> picker, the averages and the topomap stay on the same channels as the rest of
> the app. The summary line states which montage was applied (and warns if a
> channel-count mismatch prevented it).

## Where the markers come from

Every loadable session carries markers as `{"label", "sample", "t"}`:

| Source | Marker origin |
|--------|---------------|
| Native session dir | `markers.jsonl` (the markers you dropped while recording) |
| `.xdf` / `.xdfz` | every LSL **string / "Markers" stream** — each event is mapped onto the nearest EEG sample by its LSL timestamp (see [`recording/external.py`](../neurobci/recording/external.py)) |
| `.fif` | the `Raw` object's **annotations** |

So an XDF exported by LabRecorder with a separate marker stream "just works":
its event labels appear in the marker table.

> If your events are instead encoded as a *numeric trigger channel* (not a
> string marker stream), they won't appear as labels — open an issue / extend
> `_xdf_markers` to threshold that channel.

## Workflow

1. Load a recording in the **Replay** tab, then click **Load from Replay tab**.
2. The **marker table** lists every distinct label with its count. Untick
   labels you don't want. Edit the **Condition** column to *group* labels:
   give two labels the same condition name and they're pooled into one average
   (e.g. map `stim/left` and `stim/right` both to `stimulus`).
3. *(Optional)* **Combine markers** into a new derived event: tick two or more
   labels, type a name, and click **Combine ticked →**. A new marker (shown
   with a `⊕`) appears whose onsets are the *union* of the ticked markers'
   onsets (de-duplicated). You can then epoch it like any other marker; **Reset**
   drops all derived markers. This edits only the in-tab marker set — the
   recording is never modified.
5. Set the **epoch window**: `tmin`/`tmax` (s, relative to the marker),
   optional **baseline** correction interval, and optional peak-to-peak
   **amplitude rejection** (µV).
6. Choose **Preprocessing**:
   - *Configured pipeline (offline)* — reuses the platform's **existing**
     preprocessing stages (the ones from the Preprocessing tab / your profile),
     run once over the whole recording in **zero-phase / offline** mode. The
     way data are preprocessed is *not changed* here; this is the same pipeline,
     just applied offline.
   - *Raw (no preprocessing)* — epoch the signal as stored.
7. **Compute averages**.

## Outputs

- **Evoked response** — per-channel ERP overlay, one curve per condition, with
  a ±SEM band. Toggle **Butterfly** to plot every EEG channel for one
  condition.
- **Difference waveform** — tick **Difference** and pick two conditions (A − B,
  e.g. `error − correct`) to overlay their difference as a dashed curve in the
  evoked and GFP views; its ±SEM adds in quadrature. The difference is also
  selectable as a **Topo condition**, so you can map the A − B scalp
  distribution at any latency.
- **Global field power** — spatial standard deviation across EEG channels,
  one trace per condition (a reference-free measure of response strength).
- **Scalp topomap** — interpolated map of one condition's average at a chosen
  latency. It is **adaptive to the present electrodes**: only scalp EEG
  channels with a known position anchor the map, and the coloured field is
  masked to the area those electrodes actually cover (derived from their
  spacing) so it never extrapolates a value onto scalp no electrode measured.
  Name your channels with real 10-20 labels (Channels tab) for positions to
  resolve.
- **Counts** — per condition: epochs kept, total onsets, rejected, and
  out-of-bounds, plus the preprocessing description and any warnings (e.g. a
  fit-requiring stage such as ICA that was left pass-through because it was not
  calibrated).
- **Export averages (.npz)** — `times`, `sfreq`, channel names/kinds, the
  preprocessing description, and `avg__<cond>` / `sem__<cond>` / `n__<cond>`
  arrays per condition.

## Notes & caveats

- Fit-requiring stages (ICA / ASR / bad-channel interpolation) are **not**
  calibrated in this tab; if your configured pipeline contains them they pass
  through unchanged (reported as a warning). Fit them in the Preprocessing /
  Calibration flow if you need them applied.
- Averaging across a short recording yields few epochs — read the kept-epoch
  counts before trusting an ERP.
- The headless core lives in
  [`neurobci/bci/erp_analysis.py`](../neurobci/bci/erp_analysis.py) and is
  unit-tested in [`tests/test_erp_analysis.py`](../tests/test_erp_analysis.py),
  so the same analysis can be scripted without the GUI.
