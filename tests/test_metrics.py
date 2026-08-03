import unittest

from sleepvlm_bench.evaluation.metrics import classification_metrics


class MetricTests(unittest.TestCase):
    def test_invalid_predictions_remain_in_denominator(self):
        report = classification_metrics(
            ["W", "N1", "N2", "N3", "REM"],
            ["W", None, "N2", "W", "REM"],
        )
        self.assertEqual(report["n_samples"], 5)
        self.assertEqual(report["invalid_count"], 1)
        self.assertAlmostEqual(report["accuracy"], 3 / 5)
        self.assertEqual(report["confusion_matrix"]["N1"]["INVALID"], 1)

    def test_macro_and_weighted_f1_are_separate(self):
        report = classification_metrics(
            ["W", "W", "W", "W", "N1", "N2", "N3", "REM"],
            ["W", "W", "W", "W", "W", "W", "W", "W"],
        )
        self.assertNotEqual(report["macro_f1"], report["weighted_f1"])

    def test_perfect_predictions_have_unit_kappa(self):
        labels = ["W", "N1", "N2", "N3", "REM"] * 2
        report = classification_metrics(labels, labels)
        self.assertAlmostEqual(report["cohen_kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()

