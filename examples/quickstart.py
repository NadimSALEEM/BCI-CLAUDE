"""Quickstart: simulated P300 calibration -> online selection (no hardware).

    python examples/quickstart.py

Trains a P300 model on synthetic data, prints the cross-validated metrics
against chance, then runs simulated online selections and reports accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from neurobci.bci.calibration import run_simulated_calibration  # noqa: E402
from neurobci.bci.online import OnlineP300Decoder               # noqa: E402
from neurobci.config.schema import ChannelConfig                # noqa: E402
from neurobci.core.logging_setup import configure_logging       # noqa: E402
from neurobci.paradigms.base import get_paradigm                # noqa: E402
from neurobci.paradigms.synthetic import make_p300_selection_run  # noqa: E402


def main() -> int:
    configure_logging(level="WARNING", to_file=False)
    paradigm = get_paradigm("p300")
    channels = ChannelConfig()

    print("1) Calibrating (simulated P300, 300 trials)...")
    result = run_simulated_calibration(paradigm, channels, n_trials=300,
                                       model_names=["vec_lda", "riemann_lr"])
    for name, r in result.results.items():
        flag = "  <-- best" if name == result.best_model_name else ""
        print("   " + r.summary() + flag)
    if not result.is_usable:
        print("   NOT USABLE:", " ".join(result.messages))
        return 1

    print("\n2) Online selection (12 simulated runs of 6 items)...")
    decoder = OnlineP300Decoder(result.model)
    rng = np.random.default_rng(0)
    correct = 0
    for k in range(12):
        target = int(rng.integers(0, 6))
        ds, item_ids, true_item = make_p300_selection_run(
            n_items=6, n_repetitions=10, target_item=target,
            channels=channels, window=paradigm.window, seed=100 + k)
        selected, _ = decoder.decide_selection(ds.X, item_ids, 6)
        correct += int(selected == true_item)
    print(f"   selection accuracy: {correct}/12 ({100 * correct / 12:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
