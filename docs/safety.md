# Safety

NeuroBCI is a research tool. It must not be used for clinical decisions, and
it never executes arbitrary system commands.

## Command safety (the single choke-point)

Every BCI command passes through `control.router.CommandRouter.submit`, which
consults the `SafetyMonitor` and a rate limiter before any adapter runs. A
command is **blocked** (and the reason recorded) when:

- the **Emergency Stop** is active (toolbar button; latched in app state);
- the EEG **stream is not connected/stable**;
- model **confidence** is below threshold;
- **signal quality** is unacceptable;
- the command targets **external hardware** and external control is disabled
  (the default);
- the **rate limit** (commands/minute) is exceeded;
- **test mode** is on (predict but never act).

Because all paths funnel through this method, the "no action on uncertainty"
rule and the emergency stop are guaranteed for every command.

## Defaults

- **External control is disabled by default.** Only the disabled-by-default
  gate and the adapter interface exist; internal (on-screen) adapters are the
  only ones shipped, and they cannot run OS/system commands.
- The default response to an uncertain decision is **no command** (safe
  abstention), not a guess.
- Selection abstains rather than guessing when evidence is insufficient.

## Operational guidance

- Confirm **signal quality** before calibration or online use.
- Treat a *NOT USABLE* calibration as unusable — do not rely on its output.
- Keep the **Emergency Stop** reachable; it gates command execution
  immediately.
- When enabling any future external adapter, set
  `control.safety.external_control_enabled` deliberately, keep the rate limit
  conservative, and test in **test mode** first.

## Privacy

- Processing is **local by default**; there is no cloud upload and no hidden
  network transmission. The only outbound path is the **opt-in** virtual LSL
  publisher, which broadcasts on the local network when you enable it.
- Recordings store a **pseudonymous** participant id only — no directly
  identifying information by default.
- Configuration is parsed as plain JSON; **no code execution** from configs.
