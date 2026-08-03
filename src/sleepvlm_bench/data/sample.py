from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from ..constants import STAGES
from ..schema import EpochRecord


def sample_by_stage(
    records: Iterable[EpochRecord],
    targets: dict[str, int],
    *,
    split: str | None = None,
    cohort: str | None = None,
    seed: int = 2024,
) -> list[EpochRecord]:
    unknown = set(targets) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown target labels: {sorted(unknown)}")
    if any(count < 0 for count in targets.values()):
        raise ValueError("target counts must be nonnegative")

    candidates: dict[str, list[EpochRecord]] = defaultdict(list)
    for record in records:
        if split is not None and record.split != split:
            continue
        if cohort is not None and record.cohort != cohort:
            continue
        candidates[record.label].append(record)

    rng = random.Random(seed)
    selected: list[EpochRecord] = []
    for label in STAGES:
        requested = targets.get(label, 0)
        pool = sorted(candidates[label], key=lambda item: item.sample_id)
        if requested > len(pool):
            raise ValueError(
                f"requested {requested} {label} samples but only {len(pool)} are available"
            )
        selected.extend(rng.sample(pool, requested))
    return sorted(selected, key=lambda item: item.sample_id)

