#!/usr/bin/env python
"""Launch the NeuroBCI desktop application.

    python run.py [profile_name]

With no arguments it loads the 'default' configuration profile (creating
it on first run) and starts in simulation mode -- no headset required.
"""

import sys
from pathlib import Path

# Allow running from a checkout without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurobci.ui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
