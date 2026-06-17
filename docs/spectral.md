# Spectral & cognitive-state monitoring

Phase 7 adds frequency-domain analysis and *exploratory* cognitive-state
indices. The governing principle (spec §16) is **transparency**: nothing is
presented as a hidden measurement of a mental state.

## Spectral analysis (`neurobci.spectral.psd`)

- **PSD** via Welch (`compute_psd`) → `(freqs, psd[n_freqs, n_channels])`.
- **Band power**: absolute (`band_powers`) and **relative** (fraction of
  total, `relative_band_powers`) for the standard bands (delta/theta/alpha/
  beta/gamma) or any custom dict.
- **Individual alpha frequency** (`individual_alpha_frequency`): the alpha
  peak over posterior EEG channels.
- **Regional aggregation** (`regional_band_power`): mean band power per
  scalp region (frontal/central/parietal/occipital/temporal), EEG-only.

## Cognitive-state indices (`neurobci.spectral.indices`)

Three transparent ratios, each a `CognitiveIndex` carrying **everything**
needed to judge it:

| Index | Definition | Channels |
|-------|-----------|----------|
| engagement | `beta / (alpha + theta)` (Pope et al. 1995) | frontal+central+parietal |
| workload | `theta(frontal) / alpha(parietal)` (Gevins) | frontal, parietal/occipital |
| drowsiness | `(theta + alpha) / beta` | all scalp EEG |

Each index exposes: **value**, **definition** (the formula), the exact
**channels** used, the **raw band-power components** behind the ratio, a
**validity** flag (false if too few good channels / contaminated), an
optional **baseline-relative** value, and a standing **caveat**:

> *Exploratory proxy, not a direct measure of mental state. Requires
> experimental validation for your task and participant.*

A supervised cognitive-state **classifier** is intentionally **not**
shipped — that needs labelled data and would imply a confidence these
ratios don't have.

## Baselines & history (`SpectralAnalyzer`)

- `set_baseline(report)` captures current band powers + index values.
  Afterwards the topomap can show **dB change vs baseline** and indices show
  a `× baseline` ratio.
- A short **time-history** of band powers and index values feeds the
  temporal plot (sliding-window monitoring).
- **Artifact-aware**: channels flagged BAD by the quality metrics are
  excluded from indices, regional means and as topomap interpolation
  anchors.

## Topographic maps (`neurobci.spectral.topo`)

A standard normalised 10-20 layout (`POS_1020`) gives 2-D scalp positions;
`interpolate_topomap` interpolates a per-channel value vector onto the head
disc (NaN outside, bad/position-less channels excluded). The **Spectral /
State** GUI tab renders this with matplotlib (head outline + nose + sensor
markers) for a selected band and map mode (absolute / relative / baseline).

## GUI (Spectral / State tab)

PSD curve (mean over good EEG, log scale) · band-power **topomap** · a
**temporal** trace of a chosen index · a panel listing every index with its
value, formula, channels and raw components · **Capture baseline** / Clear ·
the caveat shown at all times. Requires a running stream; heavy work is
throttled to ~3 Hz so it never affects acquisition.
