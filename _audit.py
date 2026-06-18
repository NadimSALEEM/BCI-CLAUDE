"""End-to-end audit of the 'rien' XDF recordings through the full pipeline."""
import glob
import time
import numpy as np

from neurobci.recording.external import load_xdf
from neurobci.acquisition.replay_source import ReplaySource
from neurobci.acquisition.engine import AcquisitionEngine
from neurobci.config.schema import AppConfig
from neurobci.core.stream_info import KIND_EEG, StreamInfo
from neurobci.preprocessing.pipeline import Pipeline
from neurobci.preprocessing.artifacts import detect_artifacts
from neurobci.quality.metrics import compute_quality, QualityRating
from neurobci.spectral.analysis import SpectralAnalyzer
from neurobci.spectral.topo import channel_positions_2d
from collections import Counter

files = sorted(glob.glob("rien/**/*.xdf", recursive=True))
print(f"Found {len(files)} XDF file(s)\n" + "=" * 70)

for path in files:
    print(f"\n### {path}")
    t0 = time.time()
    s = load_xdf(path)
    load_s = time.time() - t0
    sf = s.sfreq
    data = s.data  # (n, ch) float32, declared uV
    info = StreamInfo(name=s.meta.get("stream_name", "x"), sfreq=sf,
                      channel_names=list(s.channel_names),
                      channel_kinds=list(s.channel_kinds), source_kind="replay")
    print(f"  load: {load_s:.2f}s | {s.n_samples} samp @ {sf:.0f}Hz "
          f"({s.n_samples/sf:.0f}s) | {len(s.channel_names)} ch | "
          f"{len(s.markers)} markers")
    print(f"  kinds: {dict(Counter(s.channel_kinds))}")
    pos, found = channel_positions_2d(s.channel_names)
    print(f"  electrode positions resolved: {int(found.sum())}/{len(s.channel_names)}")

    # --- SCALE DIAGNOSTIC (is the declared 'uV' plausible?) ---
    eeg_idx = [i for i, k in enumerate(s.channel_kinds) if k == KIND_EEG]
    X = data[:, eeg_idx].astype(np.float64)
    raw_std = np.std(X, axis=0)
    # sample-to-sample diff std removes DC/drift -> reflects true signal scale
    diff_std = np.std(np.diff(X, axis=0), axis=0)
    print(f"  scale: raw std median={np.median(raw_std):,.0f} uV | "
          f"diff(n,n-1) std median={np.median(diff_std):,.1f} uV "
          f"(EEG should be ~0.5-5)")

    # --- markers in range? ---
    if s.markers:
        samples = np.array([m["sample"] for m in s.markers])
        at0 = int(np.sum(samples == 0))
        atend = int(np.sum(samples == s.n_samples - 1))
        labels = Counter(m["label"] for m in s.markers)
        print(f"  markers: clipped@start={at0} clipped@end={atend} | "
              f"distinct labels={len(labels)} e.g. {list(labels)[:5]}")

    # --- run through engine pipeline (causal preprocessing) ---
    cfg = AppConfig()
    pipe = Pipeline.from_config(cfg.preprocessing, sf, s.channel_kinds,
                               s.channel_names)
    win = data[: int(4 * sf)]  # 4 s window
    proc = pipe.apply_window(win, "causal")
    print(f"  pipeline causal: in std={np.std(win[:,eeg_idx]):,.0f} -> "
          f"out std={np.std(proc[:,eeg_idx]):,.1f} uV")

    # --- quality on a processed window ---
    q = compute_quality(proc, info)
    ratings = Counter(c.rating.name for c in q.channels)
    print(f"  quality(processed): {dict(ratings)}")

    # --- spectral + artifacts ---
    an = SpectralAnalyzer(info, cfg.spectral)
    rep = an.analyze(proc)
    print(f"  IAF={rep.iaf:.1f}Hz | indices="
          f"{[f'{i.name}={i.value:.2f}' for i in rep.indices]}")
    ar = detect_artifacts(win, info)
    akinds = Counter(e.kind for e in ar.events if e.kind != 'blinks')
    print(f"  artifacts(raw 4s): {dict(akinds)} | blinks={ar.n_blinks}")

    # --- replay realtime smoke: does read() pace correctly? ---
    rs = ReplaySource(s, realtime=True)
    rs.start(); time.sleep(0.2); d, ts = rs.read(); rs.stop()
    exp = 0.2 * sf
    print(f"  replay 0.2s realtime: got {d.shape[0]} samp (expect ~{exp:.0f})")

print("\n" + "=" * 70 + "\nAUDIT COMPLETE")
