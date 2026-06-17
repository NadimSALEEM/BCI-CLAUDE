"""Headless smoke test: run the simulated engine for a few seconds and
print acquisition stats plus a signal-quality summary.

This needs no display and no hardware, so it is the quickest way to
confirm the data path works end-to-end:

    python scripts/engine_smoke.py [seconds]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running directly from a checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurobci.acquisition.engine import AcquisitionEngine          # noqa: E402
from neurobci.config.schema import AppConfig                       # noqa: E402
from neurobci.core.logging_setup import configure_logging          # noqa: E402
from neurobci.quality.metrics import compute_quality               # noqa: E402


def main(duration: float = 3.0) -> int:
    configure_logging(level="INFO", to_file=False)

    cfg = AppConfig()
    cfg.acquisition.source_type = "simulated"
    # Inject a couple of faults so the quality report has something to say.
    cfg.simulation.flat_channels = ["Cz"]
    cfg.simulation.noisy_channels = ["F8"]

    engine = AcquisitionEngine(cfg)
    engine.start()
    print(f"Running simulated acquisition for {duration:.1f}s ...\n")
    time.sleep(duration)

    snap = engine.state.snapshot()
    info = engine.stream_info
    print("=== Acquisition ===")
    print(f"  mode             : {snap.mode.value}")
    print(f"  connection       : {snap.connection.value}")
    print(f"  channels         : {info.n_channels} "
          f"(EEG={len(info.eeg_indices)}, EOG={len(info.eog_indices)})")
    print(f"  nominal sfreq    : {info.sfreq:.1f} Hz")
    print(f"  measured sfreq   : {snap.measured_sfreq:.1f} Hz")
    print(f"  samples received : {snap.samples_received}")
    print(f"  buffer capacity  : {engine.buffer.capacity} samples")

    data, ts = engine.latest_seconds(2.0)
    report = compute_quality(data, info)
    print("\n=== Signal quality (last 2 s) ===")
    print(f"  window           : {report.n_samples} samples")
    print(f"  overall rating   : {report.overall_rating.value}")
    print(f"  bad channels     : {report.n_bad_channels}")
    print("  per-channel:")
    for ch in report.channels:
        print(
            f"    {ch.name:>4} [{ch.kind}] {ch.rating.value:<5} "
            f"std={ch.std_uv:6.2f}uV line={ch.line_ratio*100:4.0f}% "
            f"hf={ch.hf_ratio*100:4.0f}%  -> {ch.reasons[0]}"
        )

    engine.stop()

    # Basic sanity gate so the script has a meaningful exit code.
    ok = (
        snap.samples_received > 0
        and report.n_samples > 0
        and any(c.name == "Cz" and c.rating.value == "bad" for c in report.channels)
    )
    print("\nSMOKE RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    raise SystemExit(main(secs))
