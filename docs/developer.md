# Developer guide

See [architecture.md](architecture.md) for the data-flow and module map.
This guide covers extending the platform and running validation.

## Repository map

```
neurobci/
  config/        typed schema + JSON profile manager
  core/          logging, ring buffer, stream info, app state, event bus
  acquisition/   EEGSource ABC + simulated/LSL/replay, discovery, validation,
                 LSL publisher, artifact injection, acquisition thread, engine
  preprocessing/ stages, pipeline (causal/offline), artifact detection
  quality/       explainable signal-quality metrics
  paradigms/     Paradigm ABC + registry, P300/MI/SSVEP/ErrP, synthetic data
  bci/           epoching, features, models, CCA, evaluation, calibration,
                 online, model store
  control/       decision layer, selection, safety, adapters, router, metrics,
                 mapping, ErrP correction, simulated driver
  spectral/      PSD, band power, indices, topomap, analyzer
  recording/     session writer, loader/exporter, synthetic session
  ui/            PyQt5 main window, workspaces, widgets, theme
tests/  scripts/  examples/  docs/  configs/
```

## Extension points

### Add an EEG device / source

Subclass `acquisition.base.EEGSource` (`start/read/stop/info`), return a
`StreamInfo` with correct channel kinds, and wire it into
`acquisition.engine.build_source`. Everything downstream is source-agnostic.

### Add a paradigm

Subclass `paradigms.base.Paradigm` (class labels, positive label, epoch
window, model names, `family`), decorate with `@register_paradigm`, and add a
synthetic generator + a calibration branch in
`bci.calibration._make_synthetic` (or a dedicated path like SSVEP's CCA). It
then appears automatically in the Calibration tab.

### Add a preprocessing stage

Subclass `preprocessing.stages.ProcessingStage` (define `PARAM_SPECS`,
`apply` / `process_chunk` / `apply_offline`, `validate`, `realtime_safe`) and
register it in `STAGE_REGISTRY`. The Preprocessing UI builds its parameter
editor from `describe()` automatically.

### Add a classifier

Add a builder branch in `bci.models.build_pipeline` returning an sklearn
`Pipeline` that fits on `(n_epochs, n_channels, n_times)`. Reference it from a
paradigm's `model_names`; evaluation/persistence work unchanged.

### Add a control adapter

Subclass `control.adapters.ControlAdapter` (`execute`, `is_external`). All
commands still pass through `CommandRouter.submit`, so safety gating, rate
limiting, history and metrics apply automatically. External adapters stay
disabled until `SafetyConfig.external_control_enabled` is set.

## Validation

```bash
python scripts/run_all.py              # unit tests + engine + GUI smoke
python -m unittest discover -s tests   # tests only (stdlib runner)
pytest                                 # if installed
python scripts/profile_performance.py  # real-time profiling
QT_QPA_PLATFORM=offscreen python scripts/gui_smoke.py   # headless GUI
```

## Conventions

- Tests use stdlib `unittest` (also run under pytest); synthetic generators
  provide **known ground truth** and honesty checks (chance-level on no
  signal).
- The online path is **strictly causal**; offline/non-causal code is kept
  separate and labelled.
- Config is plain dataclasses ↔ JSON (no code execution); new sections must
  round-trip through `AppConfig.from_dict`.
