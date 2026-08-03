import unittest

from sleepvlm_bench.baselines.contracts import assert_identical_test_samples
from sleepvlm_bench.data.cohorts import normalize_stage


class CohortAndContractTests(unittest.TestCase):
    def test_stage_mapping_excludes_unscored_shhs_epochs(self):
        self.assertEqual(normalize_stage("4"), "N3")
        self.assertEqual(normalize_stage("5"), "REM")
        self.assertIsNone(normalize_stage("9"))
        self.assertIsNone(normalize_stage("Movement time"))

    def test_baseline_samples_must_match_exactly(self):
        assert_identical_test_samples(["a", "b"], ["b", "a"])
        with self.assertRaises(ValueError):
            assert_identical_test_samples(["a", "b"], ["a", "c"])


if __name__ == "__main__":
    unittest.main()

