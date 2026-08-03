from __future__ import annotations

from collections.abc import Iterable


def assert_identical_test_samples(
    reference_sample_ids: Iterable[str], candidate_sample_ids: Iterable[str]
) -> None:
    reference = list(reference_sample_ids)
    candidate = list(candidate_sample_ids)
    if len(reference) != len(set(reference)) or len(candidate) != len(set(candidate)):
        raise ValueError("baseline sample lists contain duplicates")
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    if missing or unexpected:
        raise ValueError(
            f"baseline test samples differ: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

