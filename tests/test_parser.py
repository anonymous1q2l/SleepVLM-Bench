import unittest

from sleepvlm_bench.evaluation.parser import parse_stage


class ParserTests(unittest.TestCase):
    def test_first_and_last_are_distinct(self):
        text = "N1 is considered, but the final prediction is REM."
        self.assertEqual(parse_stage(text, "first").label, "N1")
        self.assertEqual(parse_stage(text, "last").label, "REM")

    def test_majority_uses_first_occurrence_as_tie_breaker(self):
        self.assertEqual(parse_stage("N2 N3 N2 REM", "majority").label, "N2")
        self.assertEqual(parse_stage("REM N2", "majority").label, "REM")

    def test_word_fragments_are_not_labels(self):
        self.assertIsNone(parse_stage("Waveforms remain nonremarkable.").label)
        self.assertIsNone(parse_stage("NREM sleep is possible.").label)

    def test_invalid_output_is_preserved(self):
        result = parse_stage("I cannot determine the stage.")
        self.assertIsNone(result.label)
        self.assertEqual(result.matches, ())


if __name__ == "__main__":
    unittest.main()

