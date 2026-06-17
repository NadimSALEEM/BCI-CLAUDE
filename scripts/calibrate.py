"""Run a simulated P300 calibration from the command line and save a model.

    python scripts/calibrate.py [n_trials] [--save]

Trains and cross-validates the configured models on synthetic P300 data
(known ground truth), prints the comparison against chance, and -- with
--save -- writes the best model under ./models. No hardware required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurobci.bci.calibration import run_simulated_calibration   # noqa: E402
from neurobci.bci.model_store import save_model                  # noqa: E402
from neurobci.config.schema import ChannelConfig                 # noqa: E402
from neurobci.core.logging_setup import configure_logging        # noqa: E402
from neurobci.paradigms.base import get_paradigm                 # noqa: E402


def main(argv: list[str]) -> int:
    configure_logging(level="WARNING", to_file=False)
    n_trials = 300
    for a in argv[1:]:
        if a.isdigit():
            n_trials = int(a)
    save = "--save" in argv

    result = run_simulated_calibration(
        get_paradigm("p300"), ChannelConfig(), sfreq=500.0,
        n_trials=n_trials, model_names=["vec_lda", "riemann_lr"],
    )

    print(f"\nCalibration on {result.n_epochs} epochs "
          f"(classes {result.class_counts}):\n")
    for name, r in result.results.items():
        flag = "  <-- best" if name == result.best_model_name else ""
        print("  " + r.summary() + flag)
    print()
    if result.is_usable:
        print(f"USABLE: best model '{result.best_model_name}' beats chance.")
    else:
        print("NOT USABLE: " + " ".join(result.messages))

    if save and result.model is not None:
        path = save_model(result.model, root="models",
                          extra_meta={"calibration": "cli"})
        print(f"\nSaved best model to {path}")
    return 0 if result.is_usable else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
