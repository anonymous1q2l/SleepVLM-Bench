import unittest

import numpy as np

from sleepvlm_bench.data.preprocess import (
    Annotation,
    PreprocessConfig,
    extract_epochs,
    preprocess_continuous,
)


class PreprocessTests(unittest.TestCase):
    def test_preprocess_and_epoch_shape(self):
        source_sfreq = 200.0
        seconds = 60
        time = np.arange(int(source_sfreq * seconds)) / source_sfreq
        signals = np.stack(
            [
                20e-6 * np.sin(2 * np.pi * 8 * time),
                30e-6 * np.sin(2 * np.pi * 1 * time),
                10e-6 * np.sin(2 * np.pi * 20 * time),
            ]
        )
        config = PreprocessConfig(powerline_hz=50.0)
        prepared = preprocess_continuous(signals, source_sfreq, config)
        self.assertEqual(prepared.shape, (3, 6000))
        self.assertTrue(np.isfinite(prepared).all())
        epochs = extract_epochs(prepared, [Annotation(0, 60, "N2")], config)
        self.assertEqual(len(epochs), 2)
        self.assertEqual(epochs[0].signals_uv.shape, (3, 3000))

    def test_unknown_annotations_are_excluded_without_shifting_time(self):
        config = PreprocessConfig(powerline_hz=None)
        signals = np.zeros((3, 9000), dtype=np.float32)
        annotations = [
            Annotation(0, 30, "W"),
            Annotation(30, 30, None),
            Annotation(60, 30, "REM"),
        ]
        epochs = extract_epochs(signals, annotations, config)
        self.assertEqual([epoch.onset_sec for epoch in epochs], [0, 60])
        self.assertEqual([epoch.source_epoch_index for epoch in epochs], [0, 2])

    def test_normalization_is_rejected(self):
        config = PreprocessConfig(normalize=True)
        with self.assertRaises(ValueError):
            config.validate(200.0)


if __name__ == "__main__":
    unittest.main()

