from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import replace
from math import floor
from typing import Iterable

from ..constants import STAGES
from ..schema import EpochRecord


DEFAULT_RATIOS = {"train": 0.7, "validation": 0.1, "test": 0.2}


def _target_sizes(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    targets = {name: floor(value) for name, value in raw.items()}
    remaining = total - sum(targets.values())
    order = sorted(ratios, key=lambda name: (raw[name] - targets[name], name), reverse=True)
    for name in order[:remaining]:
        targets[name] += 1
    return targets


def _split_cohort(
    records: list[EpochRecord], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    by_subject: dict[str, list[EpochRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject_id].append(record)

    subject_targets = _target_sizes(len(by_subject), ratios)
    label_totals = Counter(record.label for record in records)
    epoch_targets = {name: len(records) * ratio for name, ratio in ratios.items()}
    label_targets = {
        name: {label: label_totals[label] * ratio for label in STAGES}
        for name, ratio in ratios.items()
    }

    rng = random.Random(seed)
    tie_breakers = {subject: rng.random() for subject in by_subject}
    subjects = sorted(
        by_subject,
        key=lambda subject: (-len(by_subject[subject]), tie_breakers[subject], subject),
    )
    assigned_subjects = Counter()
    assigned_epochs = Counter()
    assigned_labels = {name: Counter() for name in ratios}
    assignments: dict[str, str] = {}

    for subject in subjects:
        histogram = Counter(record.label for record in by_subject[subject])
        candidates = [
            name for name in ratios if assigned_subjects[name] < subject_targets[name]
        ]
        if not candidates:
            raise RuntimeError("split capacity was exhausted before all subjects were assigned")

        def score(name: str) -> tuple[float, int, str]:
            label_error = sum(
                (
                    (assigned_labels[name][label] + histogram[label] - label_targets[name][label])
                    ** 2
                )
                / max(label_targets[name][label], 1.0)
                for label in STAGES
            )
            epoch_error = (
                assigned_epochs[name] + len(by_subject[subject]) - epoch_targets[name]
            ) ** 2 / max(epoch_targets[name], 1.0)
            return label_error + 0.25 * epoch_error, assigned_subjects[name], name

        selected = min(candidates, key=score)
        assignments[subject] = selected
        assigned_subjects[selected] += 1
        assigned_epochs[selected] += len(by_subject[subject])
        assigned_labels[selected].update(histogram)

    return assignments


def assign_subject_splits(
    records: Iterable[EpochRecord],
    ratios: dict[str, float] | None = None,
    seed: int = 2024,
) -> list[EpochRecord]:
    rows = list(records)
    split_ratios = dict(ratios or DEFAULT_RATIOS)
    if set(split_ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must contain train, validation, and test")
    if any(value <= 0 for value in split_ratios.values()):
        raise ValueError("all split ratios must be positive")
    if abs(sum(split_ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")

    by_cohort: dict[str, list[EpochRecord]] = defaultdict(list)
    for record in rows:
        by_cohort[record.cohort].append(record)

    output: list[EpochRecord] = []
    for cohort_index, cohort in enumerate(sorted(by_cohort)):
        assignments = _split_cohort(
            by_cohort[cohort], split_ratios, seed + cohort_index * 1009
        )
        output.extend(
            replace(record, split=assignments[record.subject_id])
            for record in by_cohort[cohort]
        )
    return output

