from __future__ import annotations

from collections import Counter
from typing import Iterable

from ..constants import INVALID_LABEL, STAGES


def classification_metrics(
    true_labels: Iterable[str], predicted_labels: Iterable[str | None]
) -> dict[str, object]:
    y_true = list(true_labels)
    y_pred = [label if label in STAGES else INVALID_LABEL for label in predicted_labels]
    if len(y_true) != len(y_pred):
        raise ValueError("true and predicted label lengths differ")
    if not y_true:
        raise ValueError("cannot evaluate an empty prediction set")
    invalid_true = sorted(set(y_true) - set(STAGES))
    if invalid_true:
        raise ValueError(f"ground-truth labels are invalid: {invalid_true}")

    total = len(y_true)
    correct = sum(truth == prediction for truth, prediction in zip(y_true, y_pred, strict=True))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    recall_values = []
    weighted_f1_sum = 0.0
    for label in STAGES:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        support = sum(t == label for t in y_true)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_values.append(f1)
        recall_values.append(recall)
        weighted_f1_sum += f1 * support

    true_counts = Counter(y_true)
    predicted_counts = Counter(y_pred)
    observed_agreement = correct / total
    kappa_labels = (*STAGES, INVALID_LABEL)
    expected_agreement = sum(
        (true_counts[label] / total) * (predicted_counts[label] / total)
        for label in kappa_labels
    )
    denominator = 1.0 - expected_agreement
    kappa = (observed_agreement - expected_agreement) / denominator if denominator else 0.0
    invalid_count = predicted_counts[INVALID_LABEL]

    columns = (*STAGES, INVALID_LABEL)
    confusion = {
        truth: {
            prediction: sum(
                t == truth and p == prediction
                for t, p in zip(y_true, y_pred, strict=True)
            )
            for prediction in columns
        }
        for truth in STAGES
    }
    return {
        "n_samples": total,
        "accuracy": observed_agreement,
        "macro_f1": sum(f1_values) / len(STAGES),
        "weighted_f1": weighted_f1_sum / total,
        "macro_recall": sum(recall_values) / len(STAGES),
        "cohen_kappa": kappa,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / total,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }

