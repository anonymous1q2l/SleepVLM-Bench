from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..data.manifest import read_manifest
from ..provenance import sha256_file, sha256_text, write_json
from .base import VisionLanguageBackend


def run_inference(
    *,
    backend: VisionLanguageBackend,
    manifest_path: str | Path,
    prompt_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    split: str = "test",
    prompt_id: str | None = None,
) -> Path:
    manifest_file = Path(manifest_path)
    prompt_file = Path(prompt_path)
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    records = [record for record in read_manifest(manifest_file) if record.split == split]
    if not records:
        raise ValueError(f"manifest contains no samples in split {split!r}")
    if any(not record.image_path or not record.image_sha256 for record in records):
        raise ValueError("all inference records must have an image path and SHA-256")
    for record in records:
        image_path = Path(record.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"image is missing for {record.sample_id}: {image_path}")
        actual_hash = sha256_file(image_path)
        if actual_hash != record.image_sha256:
            raise ValueError(f"image hash mismatch for {record.sample_id}")

    run_root = Path(output_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    predictions_path = run_root / "predictions.jsonl"
    metadata_path = run_root / "run.json"
    if predictions_path.exists() or metadata_path.exists():
        raise FileExistsError(f"run_id already exists and will not be overwritten: {run_id}")

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": backend.model_id,
        "model_revision": backend.model_revision,
        "backend": backend.backend_config,
        "decoding": backend.decoding_config,
        "manifest_path": str(manifest_file.resolve()),
        "manifest_sha256": sha256_file(manifest_file),
        "split": split,
        "prompt_id": prompt_id or prompt_file.stem,
        "prompt_path": str(prompt_file.resolve()),
        "prompt_sha256": sha256_text(prompt),
        "prediction_path": str(predictions_path.resolve()),
    }
    write_json(metadata_path, metadata)

    with predictions_path.open("x", encoding="utf-8") as handle:
        for record in records:
            error = None
            raw_output = ""
            try:
                raw_output = backend.predict(Path(record.image_path), prompt)
            except Exception as caught:
                error = {"type": type(caught).__name__, "message": str(caught)}
            row = {
                "schema_version": 1,
                "run_id": run_id,
                "sample_id": record.sample_id,
                "cohort": record.cohort,
                "subject_id": record.subject_id,
                "split": record.split,
                "true_label": record.label,
                "model_id": backend.model_id,
                "model_revision": backend.model_revision,
                "backend": backend.backend_config,
                "decoding": backend.decoding_config,
                "prompt_id": metadata["prompt_id"],
                "prompt_sha256": metadata["prompt_sha256"],
                "image_path": record.image_path,
                "image_sha256": record.image_sha256,
                "raw_output": raw_output,
                "error": error,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return predictions_path
