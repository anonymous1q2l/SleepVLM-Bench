from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..schema import MANIFEST_COLUMNS, EpochRecord


def read_manifest(path: str | Path) -> list[EpochRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        records = [EpochRecord.from_row(row) for row in reader]
    validate_manifest(records)
    return records


def write_manifest(records: Iterable[EpochRecord], path: str | Path) -> None:
    rows = sorted(records, key=lambda item: (item.cohort, item.subject_id, item.onset_sec))
    validate_manifest(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(record.to_row() for record in rows)
    temporary_path.replace(output_path)


def validate_manifest(records: Iterable[EpochRecord]) -> None:
    sample_ids: set[str] = set()
    subject_splits: dict[tuple[str, str], str] = {}
    for record in records:
        record.validate()
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        sample_ids.add(record.sample_id)
        if not record.split:
            continue
        subject_key = (record.cohort, record.subject_id)
        previous = subject_splits.setdefault(subject_key, record.split)
        if previous != record.split:
            raise ValueError(
                f"subject leakage: {subject_key} appears in {previous} and {record.split}"
            )


def summarize_manifest(records: Iterable[EpochRecord]) -> dict[str, object]:
    rows = list(records)
    return {
        "samples": len(rows),
        "subjects": len({(row.cohort, row.subject_id) for row in rows}),
        "by_cohort": dict(sorted(Counter(row.cohort for row in rows).items())),
        "by_label": dict(sorted(Counter(row.label for row in rows).items())),
        "by_split": dict(sorted(Counter(row.split or "unassigned" for row in rows).items())),
        "by_cohort_label": {
            f"{cohort}:{label}": count
            for (cohort, label), count in sorted(
                Counter((row.cohort, row.label) for row in rows).items()
            )
        },
    }

