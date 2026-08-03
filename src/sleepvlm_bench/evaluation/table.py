from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def reports_to_csv(report_paths: Iterable[str | Path], output_path: str | Path) -> None:
    rows = []
    for report_path in report_paths:
        path = Path(report_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        identity = report.get("run", {})
        required_identity = {"run_id", "model_id", "prompt_id", "prompt_sha256"}
        missing_identity = {
            field for field in required_identity if not identity.get(field)
        }
        if missing_identity:
            raise ValueError(
                f"report {path} is missing run identity: {sorted(missing_identity)}"
            )
        for cohort, metrics in report.get("metrics_by_cohort", {}).items():
            rows.append(
                {
                    "run_id": identity.get("run_id", ""),
                    "model_id": identity.get("model_id", ""),
                    "model_revision": identity.get("model_revision", ""),
                    "prompt_id": identity.get("prompt_id", ""),
                    "parser_mode": report["parser_mode"],
                    "cohort": cohort,
                    "n_samples": metrics["n_samples"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "macro_recall": metrics["macro_recall"],
                    "cohen_kappa": metrics["cohen_kappa"],
                    "invalid_count": metrics["invalid_count"],
                    "report_path": str(path.resolve()),
                }
            )
    if not rows:
        raise ValueError("no cohort metrics were found in the supplied reports")
    rows.sort(key=lambda row: (row["cohort"], row["model_id"], row["prompt_id"]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
