import tempfile
import unittest
from pathlib import Path

from sleepvlm_bench.constants import STAGES
from sleepvlm_bench.data.manifest import read_manifest, validate_manifest, write_manifest
from sleepvlm_bench.data.sample import sample_by_stage
from sleepvlm_bench.data.split import assign_subject_splits
from sleepvlm_bench.schema import EpochRecord


def make_records(subject_count=20):
    records = []
    artifact_index = 0
    for subject_index in range(subject_count):
        for stage_index, label in enumerate(STAGES):
            records.append(
                EpochRecord(
                    sample_id=f"TEST:s{subject_index:03d}:{stage_index:03d}",
                    cohort="TEST",
                    subject_id=f"s{subject_index:03d}",
                    recording_id=f"r{subject_index:03d}",
                    epoch_index=stage_index,
                    onset_sec=stage_index * 30.0,
                    label=label,
                    source_psg_path="/raw/test.edf",
                    source_annotation_path="/raw/test.xml",
                    source_sfreq=200.0,
                    artifact_path="/prepared/test.npz",
                    artifact_index=artifact_index,
                )
            )
            artifact_index += 1
    return records


class ManifestSplitSampleTests(unittest.TestCase):
    def test_split_is_deterministic_and_subject_disjoint(self):
        records = make_records()
        first = assign_subject_splits(records, seed=2024)
        second = assign_subject_splits(records, seed=2024)
        self.assertEqual(
            [(row.sample_id, row.split) for row in first],
            [(row.sample_id, row.split) for row in second],
        )
        validate_manifest(first)
        subject_splits = {}
        for row in first:
            subject_splits.setdefault(row.subject_id, set()).add(row.split)
        self.assertTrue(all(len(splits) == 1 for splits in subject_splits.values()))
        self.assertEqual(
            {split: sum(next(iter(values)) == split for values in subject_splits.values())
             for split in ("train", "validation", "test")},
            {"train": 14, "validation": 2, "test": 4},
        )

    def test_balanced_sampling_is_exact_and_deterministic(self):
        split_records = assign_subject_splits(make_records(), seed=2024)
        targets = {label: 2 for label in STAGES}
        first = sample_by_stage(split_records, targets, split="test", seed=7)
        second = sample_by_stage(split_records, targets, split="test", seed=7)
        self.assertEqual([row.sample_id for row in first], [row.sample_id for row in second])
        for label in STAGES:
            self.assertEqual(sum(row.label == label for row in first), 2)

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            records = assign_subject_splits(make_records(5), seed=11)
            write_manifest(records, path)
            self.assertEqual(read_manifest(path), sorted(
                records, key=lambda item: (item.cohort, item.subject_id, item.onset_sec)
            ))


if __name__ == "__main__":
    unittest.main()

