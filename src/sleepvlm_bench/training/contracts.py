from __future__ import annotations

from pathlib import Path


REQUIRED_LORA_ARTIFACTS = (
    "resolved_config.json",
    "train_log.jsonl",
    "validation_metrics.json",
    "test_predictions.jsonl",
    "test_metrics.json",
    "adapter_config.json",
)


def validate_lora_artifacts(run_directory: str | Path) -> list[str]:
    root = Path(run_directory)
    return [name for name in REQUIRED_LORA_ARTIFACTS if not (root / name).is_file()]


def assert_three_seed_runs(experiment_directory: str | Path) -> None:
    root = Path(experiment_directory)
    failures = {}
    for seed in (2024, 2025, 2026):
        missing = validate_lora_artifacts(root / f"seed_{seed}")
        if missing:
            failures[str(seed)] = missing
    if failures:
        raise ValueError(f"LoRA experiment is incomplete: {failures}")

