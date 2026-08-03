from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset: dict[str, Any]
    preprocessing: dict[str, Any]
    paper_diagnostic_counts: dict[str, int]


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = load_toml(path)
    required = {"dataset", "preprocessing", "paper_diagnostic_counts"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"dataset config is missing sections: {sorted(missing)}")
    return DatasetConfig(
        dataset=dict(raw["dataset"]),
        preprocessing=dict(raw["preprocessing"]),
        paper_diagnostic_counts={
            str(label): int(count) for label, count in raw["paper_diagnostic_counts"].items()
        },
    )
