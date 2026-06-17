# Installation

## Requirements

- Python ≥ 3.10 (validated on **3.14**)
- The packages in [`requirements.txt`](../requirements.txt): numpy, scipy,
  scikit-learn, mne, pyriemann, **pylsl** (needs native `liblsl`), **PyQt5**,
  pyqtgraph, matplotlib.

## Install

```bash
python -m pip install -r requirements.txt
# or, as an editable package (adds the `neurobci` console script):
python -m pip install -e .
# developer extras (pytest, coverage):
python -m pip install -r requirements-dev.txt
```

## liblsl (only for live acquisition / virtual LSL)

`pylsl` needs the native **liblsl** shared library at runtime. It is bundled
with recent `pylsl` wheels; if `import pylsl` fails to find it, install
liblsl from the LSL project and ensure it is on the library path. Simulation
and replay modes do **not** require liblsl.

## Verify the install

```bash
python scripts/run_all.py            # unit tests + engine smoke + GUI smoke
python scripts/profile_performance.py
python examples/quickstart.py        # sim calibrate -> online selection
```

All three should report PASS / sensible numbers with no hardware.

## Run

```bash
python run.py                # default profile, simulation mode
python -m neurobci           # same, as a module
python run.py examples/configs/motor_imagery   # a specific profile
```

## Windows note (python launcher)

If `python` opens the Microsoft Store, the App-Execution alias is shadowing
the interpreter. Call it by full path (e.g.
`C:\Users\<you>\AppData\Local\Python\pythoncore-3.14-64\python.exe`) or
disable the alias in *Settings → Apps → Advanced app settings → App execution
aliases*.
