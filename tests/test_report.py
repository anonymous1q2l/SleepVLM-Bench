import json
import tempfile
import unittest
from pathlib import Path

from sleepvlm_bench.evaluation.report import evaluate_prediction_file
from sleepvlm_bench.evaluation.table import reports_to_csv
from sleepvlm_bench.provenance import write_jsonl


class ReportTests(unittest.TestCase):
    def test_report_keeps_invalid_and_groups_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            common = {
                "run_id": "test_run",
                "model_id": "test/model",
                "model_revision": "abc123",
                "prompt_id": "image_only",
                "prompt_sha256": "deadbeef",
            }
            write_jsonl(
                predictions,
                [
                    {
                        **common,
                        "sample_id": "a",
                        "cohort": "A",
                        "true_label": "W",
                        "raw_output": "The answer is W.",
                        "error": None,
                    },
                    {
                        **common,
                        "sample_id": "b",
                        "cohort": "B",
                        "true_label": "N1",
                        "raw_output": "Unable to decide.",
                        "error": None,
                    },
                ],
            )
            report_path = root / "metrics.json"
            parsed_path = root / "parsed.jsonl"
            report = evaluate_prediction_file(
                predictions, report_path, parsed_output_path=parsed_path
            )
            self.assertEqual(report["metrics"]["invalid_count"], 1)
            self.assertEqual(set(report["metrics_by_cohort"]), {"A", "B"})
            self.assertEqual(report["run"]["model_id"], "test/model")
            parsed = [json.loads(line) for line in parsed_path.read_text().splitlines()]
            self.assertIsNone(parsed[1]["predicted_label"])

            table_path = root / "table.csv"
            reports_to_csv([report_path], table_path)
            self.assertEqual(len(table_path.read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()

