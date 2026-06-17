"""Generate a recorded session on disk with known ground truth.

Runs the simulator deterministically, injecting a P300 on *target* stimuli,
and writes a real session (EEG + timestamps + markers + metadata) via the
normal :class:`SessionRecorder`. The result feeds replay and regression
tests: simulate -> record -> replay -> decode should recover the planted
targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobci.acquisition.simulated import SimulatedSource
from neurobci.config.schema import AppConfig, ChannelConfig, SimulationConfig
from neurobci.recording.writer import SessionRecorder


def record_p300_session(
    out_dir: Path | str,
    n_stimuli: int = 200,
    target_ratio: float = 0.25,
    isi_s: float = 0.4,
    sfreq: float = 500.0,
    channels: ChannelConfig | None = None,
    p300_amp_uv: float = 12.0,
    participant: str = "sim",
    seed: int = 0,
) -> Path:
    """Write a synthetic P300 session and return its directory."""

    channels = channels or ChannelConfig()
    sim = SimulationConfig(seed=seed, blink_rate_hz=0.15)
    src = SimulatedSource(channels, sim, sfreq=sfreq, realtime=False)
    src.start()

    config = AppConfig()
    config.acquisition.expected_sfreq = sfreq
    rec = SessionRecorder(out_dir, src.info, config, participant_id=participant,
                          notes="synthetic P300 ground-truth session")
    rec.start()

    rng = np.random.default_rng(seed)
    isi = int(isi_s * sfreq)
    try:
        for _ in range(n_stimuli):
            is_target = rng.random() < target_ratio
            onset = rec.n_samples
            if is_target:
                src.inject_erp(p300_amp_uv)
            data, ts = src.generate(isi)
            rec.push_marker("target" if is_target else "nontarget",
                            timestamp=float(ts[0]), sample=onset)
            rec.write(data, ts)
        # A little tail so the last epoch is fully contained.
        data, ts = src.generate(int(0.6 * sfreq))
        rec.write(data, ts)
    finally:
        rec.stop()
    return rec.path
