from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import classification_metrics
from .parser import parse_stage
from ..provenance import read_jsonl, sha256_file, write_json, write_jsonl


def evaluate_prediction_file(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    parser_mode: str = "first",
    parsed_output_path: str | Path | None = None,
) -> dict[str, object]:
    rows = read_jsonl(predictions_path)
    sample_ids: set[str] = set()
    parsed_rows = []
    true_labels = []
    predicted_labels = []
    cohort_labels: dict[str, tuple[list[str], list[str | None]]] = defaultdict(
        lambda: ([], [])
    )
    identity_fields = ("run_id", "model_id", "model_revision", "prompt_id", "prompt_sha256")
    identity: dict[str, object] = {}
    for index, row in enumerate(rows, start=1):
        required = {"sample_id", "cohort", "true_label", "raw_output"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"prediction row {index} is missing fields: {sorted(missing)}")
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        sample_ids.add(sample_id)
        parsed = parse_stage(
            None if row.get("error") else str(row.get("raw_output") or ""), parser_mode
        )
        true_label = str(row["true_label"])
        cohort = str(row["cohort"])
        true_labels.append(true_label)
        predicted_labels.append(parsed.label)
        cohort_labels[cohort][0].append(true_label)
        cohort_labels[cohort][1].append(parsed.label)
        for field in identity_fields:
            if field not in row:
                continue
            if field in identity and identity[field] != row[field]:
                raise ValueError(f"prediction file mixes multiple values for {field}")
            identity[field] = row[field]
        parsed_rows.append(
            {
                **row,
                "parser_mode": parser_mode,
                "parser_matches": list(parsed.matches),
                "predicted_label": parsed.label,
            }
        )

    metrics = classification_metrics(true_labels, predicted_labels)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_file": str(Path(predictions_path).resolve()),
        "prediction_file_sha256": sha256_file(predictions_path),
        "parser_mode": parser_mode,
        "run": identity,
        "metrics": metrics,
        "metrics_by_cohort": {
            cohort: classification_metrics(labels[0], labels[1])
            for cohort, labels in sorted(cohort_labels.items())
        },
    }
    write_json(output_path, report)
    if parsed_output_path is not None:
        write_jsonl(parsed_output_path, parsed_rows)
    return report
