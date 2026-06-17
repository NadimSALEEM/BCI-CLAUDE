"""StreamInfo channel bookkeeping and validation."""

import unittest

from neurobci.core.stream_info import KIND_EEG, KIND_EOG, StreamInfo


class TestStreamInfo(unittest.TestCase):
    def _info(self):
        return StreamInfo(
            name="test",
            sfreq=500.0,
            channel_names=["Fp1", "O1", "EOG"],
            channel_kinds=[KIND_EEG, KIND_EEG, KIND_EOG],
        )

    def test_indices(self):
        info = self._info()
        self.assertEqual(info.n_channels, 3)
        self.assertEqual(info.eeg_indices, [0, 1])
        self.assertEqual(info.eog_indices, [2])
        self.assertEqual(info.index_of("O1"), 1)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            StreamInfo("t", 500.0, ["a", "b"], [KIND_EEG])

    def test_bad_sfreq_raises(self):
        with self.assertRaises(ValueError):
            StreamInfo("t", 0.0, ["a"], [KIND_EEG])


if __name__ == "__main__":
    unittest.main()
