"""Name-based electrode/montage auto-classification."""

import unittest

from neurobci.core.electrodes import (
    classify_channel,
    classify_channels,
    is_frontal,
    region_of,
)
from neurobci.core.stream_info import KIND_EEG, KIND_EOG, KIND_MISC


class TestClassifyByName(unittest.TestCase):
    def test_scalp_electrodes_across_montages(self):
        # 10-20 and extended 10-10 names, midline and numbered, any case.
        for name in ("Fp1", "Fpz", "AF7", "Fz", "F7", "FC5", "FCz", "Cz",
                     "C3", "T7", "T8", "TP9", "CP3", "Pz", "P4", "PO8",
                     "POz", "O1", "Oz", "Iz", "fp2", "fc6"):
            self.assertEqual(classify_channel(name), KIND_EEG, name)

    def test_eog_by_name(self):
        for name in ("EOG", "VEOG", "HEOG", "EOGv", "left-EOG"):
            self.assertEqual(classify_channel(name), KIND_EOG, name)

    def test_misc_by_name(self):
        for name in ("ECG", "EKG1", "EMG_left", "GSR", "RESP", "STI014",
                     "TRIGGER", "Status", "ACC_X", "A1", "M2", "REF"):
            self.assertEqual(classify_channel(name), KIND_MISC, name)

    def test_declared_type_overrides_blanket_eeg(self):
        # A file may type an ocular/cardiac lead as a blanket "EEG".
        self.assertEqual(classify_channel("EOG", "EEG"), KIND_EOG)
        self.assertEqual(classify_channel("X1", "ecg"), KIND_MISC)
        self.assertEqual(classify_channel("VEOG", "eog"), KIND_EOG)

    def test_name_overrides_vague_or_blanket_type(self):
        # Blanket "EEG" type on a clearly-ocular/aux name -> name wins.
        self.assertEqual(classify_channel("ECG", "EEG"), KIND_MISC)
        self.assertEqual(classify_channel("Cz", "misc"), KIND_EEG)

    def test_unknown_labels_default_to_eeg(self):
        for name in ("Ch1", "E17", "channel_3"):
            self.assertEqual(classify_channel(name), KIND_EEG, name)

    def test_classify_channels_vectorised(self):
        names = ["Fp1", "Cz", "EOG", "ECG"]
        types = ["EEG", "EEG", "EEG", "EEG"]
        self.assertEqual(classify_channels(names, types),
                         [KIND_EEG, KIND_EEG, KIND_EOG, KIND_MISC])

    def test_region_and_frontal(self):
        self.assertEqual(region_of("Fp1"), "frontal")
        self.assertEqual(region_of("O2"), "occipital")
        self.assertTrue(is_frontal("AF3"))
        self.assertFalse(is_frontal("Pz"))


if __name__ == "__main__":
    unittest.main()
