"""Stream-vs-montage validation logic."""

import unittest

from neurobci.acquisition.validation import Severity, validate_stream
from neurobci.config.schema import ChannelConfig
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo


def _info(names, kinds, sfreq=500.0):
    return StreamInfo("t", sfreq, names, kinds)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.expected = ChannelConfig(
            eeg_channels=["F3", "F4", "Cz"], eog_channels=["EOG"]
        )

    def _matching_info(self, sfreq=500.0):
        return _info(
            ["F3", "F4", "Cz", "EOG"],
            [KIND_EEG, KIND_EEG, KIND_EEG, KIND_EOG],
            sfreq,
        )

    def test_perfect_match_is_valid(self):
        report = validate_stream(self._matching_info(), self.expected, 500.0)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.errors, [])

    def test_sfreq_mismatch_is_error(self):
        report = validate_stream(self._matching_info(sfreq=250.0), self.expected, 500.0)
        self.assertFalse(report.is_valid)
        self.assertTrue(any(i.code == "sfreq_mismatch" for i in report.errors))

    def test_count_mismatch_is_error(self):
        info = _info(["F3", "F4"], [KIND_EEG, KIND_EEG])
        report = validate_stream(info, self.expected, 500.0)
        self.assertFalse(report.is_valid)
        self.assertTrue(any(i.code == "count_mismatch" for i in report.errors))

    def test_order_mismatch_is_warning_only(self):
        info = _info(
            ["F4", "F3", "Cz", "EOG"],
            [KIND_EEG, KIND_EEG, KIND_EEG, KIND_EOG],
        )
        report = validate_stream(info, self.expected, 500.0)
        self.assertTrue(report.is_valid)  # warning, not error
        self.assertTrue(any(i.code == "order_mismatch" for i in report.warnings))

    def test_missing_eog_warns(self):
        info = _info(
            ["F3", "F4", "Cz", "EOG"],
            [KIND_EEG, KIND_EEG, KIND_EEG, KIND_EEG],  # EOG mislabelled as EEG
        )
        report = validate_stream(info, self.expected, 500.0)
        self.assertTrue(any(i.code == "eog_missing" for i in report.warnings))

    def test_duplicate_names_is_error(self):
        info = _info(
            ["F3", "F3", "Cz", "EOG"],
            [KIND_EEG, KIND_EEG, KIND_EEG, KIND_EOG],
        )
        report = validate_stream(info, self.expected, 500.0)
        self.assertFalse(report.is_valid)
        self.assertTrue(any(i.code == "duplicate_names" for i in report.errors))


if __name__ == "__main__":
    unittest.main()
