"""Artifact detection on signals with known defects."""

import unittest

import numpy as np

from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo
from neurobci.preprocessing.artifacts import Severity, detect_artifacts

SF = 500.0


def _info(names, kinds):
    return StreamInfo("t", SF, names, kinds)


class TestArtifacts(unittest.TestCase):
    def setUp(self):
        self.n = 1500
        self.t = np.arange(self.n) / SF
        self.rng = np.random.default_rng(0)

    def _alpha(self, amp=10.0):
        return amp * np.sin(2 * np.pi * 10 * self.t) + self.rng.normal(0, 3, self.n)

    def test_flat_channel_rejected(self):
        data = np.column_stack([self._alpha(), 0.05 * self.rng.normal(0, 1, self.n)])
        rep = detect_artifacts(data, _info(["O1", "Cz"], [KIND_EEG, KIND_EEG]))
        self.assertIn("Cz", rep.bad_channels)
        kinds = {e.kind for e in rep.for_channel("Cz")}
        self.assertIn("flat", kinds)

    def test_clipping_rejected(self):
        clipped = np.clip(300 * np.sin(2 * np.pi * 3 * self.t), -250, 250)
        data = np.column_stack([self._alpha(), clipped])
        rep = detect_artifacts(data, _info(["O1", "P3"], [KIND_EEG, KIND_EEG]))
        self.assertIn("P3", rep.bad_channels)

    def test_gradient_pop_warned(self):
        x = self._alpha()
        x[700] += 300.0  # electrode pop
        rep = detect_artifacts(x[:, None], _info(["F3"], [KIND_EEG]))
        kinds = {e.kind for e in rep.for_channel("F3")}
        self.assertIn("gradient", kinds)

    def test_muscle_warned(self):
        emg = self.rng.normal(0, 15, self.n)  # broadband -> lots of HF
        rep = detect_artifacts(emg[:, None], _info(["T7"], [KIND_EEG]))
        kinds = {e.kind for e in rep.for_channel("T7")}
        self.assertIn("muscle", kinds)

    def test_line_noise_warned(self):
        x = 8 * np.sin(2 * np.pi * 10 * self.t) + 30 * np.sin(2 * np.pi * 50 * self.t)
        rep = detect_artifacts(x[:, None], _info(["Pz"], [KIND_EEG]))
        kinds = {e.kind for e in rep.for_channel("Pz")}
        self.assertIn("line_noise", kinds)

    def test_blinks_counted_on_eog(self):
        eog = self.rng.normal(0, 3, self.n)
        for onset in (200, 600, 1000):       # three blinks
            eog[onset:onset + 40] += 100.0
        rep = detect_artifacts(eog[:, None], _info(["EOG"], [KIND_EOG]))
        self.assertEqual(rep.n_blinks, 3)
        # Per-channel blink event is emitted and isolated from artifacts.
        per_ch = [e for e in rep.blink_events if e.scope == "channel"]
        self.assertEqual(len(per_ch), 1)
        self.assertEqual(per_ch[0].channel, "EOG")
        self.assertNotIn("blinks", {e.kind for e in rep.events
                                    if e.kind != "blinks"})

    def test_blinks_fall_back_to_frontal_eeg_without_eog(self):
        # No EOG channel: blinks should be picked up on a frontal electrode,
        # not on an occipital one.
        fp = self.rng.normal(0, 3, self.n)
        for onset in (300, 900):
            fp[onset:onset + 40] += 100.0
        occ = self._alpha()
        rep = detect_artifacts(np.column_stack([fp, occ]),
                               _info(["Fp1", "O1"], [KIND_EEG, KIND_EEG]))
        self.assertEqual(rep.n_blinks, 2)
        channels = {e.channel for e in rep.blink_events if e.scope == "channel"}
        self.assertEqual(channels, {"Fp1"})

    def test_no_blink_source_emits_no_blink_events(self):
        # Occipital-only montage, no EOG: nothing to detect blinks on.
        rep = detect_artifacts(self._alpha()[:, None], _info(["O1"], [KIND_EEG]))
        self.assertEqual(rep.blink_events, [])

    def test_missing_samples_global(self):
        x = self._alpha()
        x[10] = np.nan
        rep = detect_artifacts(x[:, None], _info(["O2"], [KIND_EEG]))
        self.assertTrue(any(e.kind == "missing_samples" for e in rep.events))

    def test_clean_signal_no_rejects(self):
        data = np.column_stack([self._alpha(), self._alpha()])
        rep = detect_artifacts(data, _info(["O1", "O2"], [KIND_EEG, KIND_EEG]))
        self.assertEqual(rep.bad_channels, [])

    def test_large_dc_offset_not_flagged(self):
        # Unfiltered EEG often sits on a big baseline (e.g. +2000 uV). That is
        # not clipping/amplitude — only the AC signal should be judged.
        data = np.column_stack([self._alpha() + 2000.0, self._alpha() - 1500.0])
        rep = detect_artifacts(data, _info(["O1", "O2"], [KIND_EEG, KIND_EEG]))
        flagged = {e.kind for e in rep.events}
        self.assertNotIn("clipping", flagged)
        self.assertNotIn("amplitude", flagged)
        self.assertEqual(rep.bad_channels, [])


if __name__ == "__main__":
    unittest.main()
