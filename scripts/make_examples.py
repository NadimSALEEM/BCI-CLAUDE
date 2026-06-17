"""Generate example configuration profiles (and optionally example data).

    python scripts/make_examples.py            # writes examples/configs/*.json
    python scripts/make_examples.py --data     # also writes example sessions+model

Configs are small and committed; data/models are generated on demand (they
are git-ignored) so the repository stays light.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurobci.config.manager import ConfigManager           # noqa: E402
from neurobci.config.schema import AppConfig                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "examples" / "configs"
DATA_DIR = ROOT / "examples" / "data"


def _profile(name: str, paradigm: str, stages: list[dict]) -> AppConfig:
    cfg = AppConfig(profile_name=name, paradigm=paradigm)
    cfg.preprocessing.stages = stages
    return cfg


def write_configs() -> None:
    mgr = ConfigManager(CONFIG_DIR)
    causal = "causal"
    profiles = {
        "p300": _profile("p300", "p300", [
            {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 0.5, "order": 4}},
            {"type": "notch", "enabled": True, "params": {"freq_hz": 50.0, "quality": 30.0}},
            {"type": "lowpass", "enabled": True, "params": {"cutoff_hz": 20.0, "order": 4}},
            {"type": "car", "enabled": True, "params": {}},
        ]),
        "motor_imagery": _profile("motor_imagery", "motor_imagery", [
            {"type": "bandpass", "enabled": True,
             "params": {"low_hz": 8.0, "high_hz": 30.0, "order": 4}},
            {"type": "car", "enabled": True, "params": {}},
        ]),
        "ssvep": _profile("ssvep", "ssvep", [
            {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 3.0, "order": 4}},
            {"type": "notch", "enabled": True, "params": {"freq_hz": 50.0, "quality": 30.0}},
            {"type": "lowpass", "enabled": True, "params": {"cutoff_hz": 45.0, "order": 4}},
        ]),
        "errp": _profile("errp", "errp", [
            {"type": "highpass", "enabled": True, "params": {"cutoff_hz": 1.0, "order": 4}},
            {"type": "notch", "enabled": True, "params": {"freq_hz": 50.0, "quality": 30.0}},
            {"type": "lowpass", "enabled": True, "params": {"cutoff_hz": 10.0, "order": 4}},
            {"type": "car", "enabled": True, "params": {}},
        ]),
    }
    for cfg in profiles.values():
        path = mgr.save(cfg)
        print(f"wrote {path}")


def write_data() -> None:
    from neurobci.bci.calibration import run_simulated_calibration
    from neurobci.bci.model_store import save_model
    from neurobci.config.schema import ChannelConfig
    from neurobci.paradigms.base import get_paradigm
    from neurobci.recording.synthetic_session import record_p300_session

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sess = record_p300_session(DATA_DIR, n_stimuli=200, p300_amp_uv=12.0, seed=0)
    print(f"wrote example session {sess}")
    res = run_simulated_calibration(get_paradigm("p300"), ChannelConfig(),
                                    n_trials=300, model_names=["vec_lda"])
    if res.model is not None:
        path = save_model(res.model, root=DATA_DIR / "models",
                          extra_meta={"calibration": "example"})
        print(f"wrote example model {path}")


def main(argv: list[str]) -> int:
    write_configs()
    if "--data" in argv:
        write_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
