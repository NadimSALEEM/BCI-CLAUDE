"""Performance profiling: is the platform comfortably real-time?

Measures acquisition rate, preprocessing throughput, single-epoch inference
latency and spectral-analysis time, and prints real-time factors. No
hardware required.

    python scripts/profile_performance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

SF = 500.0


def _time(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def profile_acquisition() -> tuple[float, float]:
    from neurobci.acquisition.engine import AcquisitionEngine
    from neurobci.config.schema import AppConfig
    eng = AcquisitionEngine(AppConfig())
    eng.start()
    time.sleep(1.5)
    snap = eng.state.snapshot()
    eng.stop()
    return snap.measured_sfreq, snap.measured_sfreq / SF


def profile_preprocessing() -> tuple[float, float]:
    from neurobci.config.schema import PreprocessingConfig
    from neurobci.core.stream_info import KIND_EEG
    from neurobci.preprocessing.pipeline import Pipeline
    kinds = [KIND_EEG] * 20
    pipe = Pipeline.from_config(PreprocessingConfig(), SF, kinds)
    chunk = np.random.default_rng(0).standard_normal((int(0.05 * SF), 20))
    reps = 2000
    per = _time(lambda: pipe.process_chunk(chunk), reps)
    throughput = chunk.shape[0] / per
    return per * 1000, throughput / SF


def profile_inference() -> tuple[float, float]:
    from neurobci.bci.calibration import run_simulated_calibration
    from neurobci.config.schema import ChannelConfig
    from neurobci.paradigms.base import get_paradigm
    from neurobci.paradigms.synthetic import make_p300_dataset
    res = run_simulated_calibration(get_paradigm("p300"), ChannelConfig(),
                                    n_trials=150, model_names=["vec_lda"])
    ds = make_p300_dataset(n_trials=1, channels=ChannelConfig(),
                           window=get_paradigm("p300").window, seed=5)
    epoch = ds.X[:1]
    per = _time(lambda: res.model.predict_proba(epoch), 500)
    return per * 1000, per * 1000


def profile_spectral() -> float:
    from neurobci.acquisition.default_montages import build_stream_info
    from neurobci.config.schema import ChannelConfig, SpectralConfig
    from neurobci.spectral.analysis import SpectralAnalyzer
    info = build_stream_info(ChannelConfig(), SF, "simulated")
    an = SpectralAnalyzer(info, SpectralConfig())
    data = np.random.default_rng(0).standard_normal((int(4 * SF), info.n_channels))
    return _time(lambda: an.analyze(data), 20) * 1000


def main() -> int:
    print("Profiling (no hardware)...\n")
    acq_rate, acq_factor = profile_acquisition()
    pre_ms, pre_factor = profile_preprocessing()
    inf_ms, _ = profile_inference()
    spec_ms = profile_spectral()

    print(f"{'stage':<26}{'metric':<22}{'value'}")
    print("-" * 60)
    print(f"{'acquisition (sim)':<26}{'effective rate':<22}{acq_rate:6.1f} Hz "
          f"({acq_factor:.2f}x nominal)")
    print(f"{'preprocessing (20ch)':<26}{'per 50 ms chunk':<22}{pre_ms:6.3f} ms "
          f"({pre_factor:.0f}x real-time)")
    print(f"{'inference (P300 epoch)':<26}{'predict_proba':<22}{inf_ms:6.3f} ms/epoch")
    print(f"{'spectral (4 s window)':<26}{'analyze':<22}{spec_ms:6.2f} ms")

    ok = (acq_factor > 0.9 and pre_factor > 5 and inf_ms < 50 and spec_ms < 300)
    print("\nPROFILE:", "PASS (comfortably real-time)" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
