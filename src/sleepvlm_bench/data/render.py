from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from ..schema import EpochRecord
from ..provenance import sha256_file


def load_epoch_signals(record: EpochRecord) -> np.ndarray:
    with np.load(record.artifact_path, allow_pickle=False) as archive:
        signals = np.asarray(archive["signals"][record.artifact_index], dtype=np.float32)
        unit = str(archive["unit"])
    if signals.ndim != 2 or signals.shape[0] != 3:
        raise ValueError(f"invalid signal shape for {record.sample_id}: {signals.shape}")
    if unit != "uV":
        raise ValueError(f"unexpected unit for {record.sample_id}: {unit}")
    return signals


def render_epoch(
    signals_uv: np.ndarray,
    output_path: str | Path,
    *,
    sfreq: float = 100.0,
    epoch_seconds: int = 30,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(signals_uv)
    expected_samples = int(round(sfreq * epoch_seconds))
    if values.shape != (3, expected_samples):
        raise ValueError(f"expected shape (3, {expected_samples}), got {values.shape}")
    time_axis = np.arange(expected_samples, dtype=np.float64) / sfreq
    labels = ("EEG", "EOG", "EMG")

    figure, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for axis, channel, channel_values in zip(axes, labels, values, strict=True):
        axis.plot(time_axis, channel_values, color="#111111", linewidth=0.65)
        axis.set_ylabel(f"{channel}\n(uV)")
        axis.grid(True, axis="x", color="#b8b8b8", linewidth=0.5)
        axis.margins(x=0)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, epoch_seconds)
    axes[-1].set_xticks(np.arange(0, epoch_seconds + 1, 5))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, facecolor="white")
    plt.close(figure)


def render_manifest(
    records: Iterable[EpochRecord], output_root: str | Path
) -> list[EpochRecord]:
    root = Path(output_root)
    rendered = []
    for record in records:
        safe_id = record.sample_id.replace(":", "__")
        output_path = root / record.cohort.lower() / f"{safe_id}.png"
        render_epoch(load_epoch_signals(record), output_path)
        rendered.append(
            replace(
                record,
                image_path=str(output_path.resolve()),
                image_sha256=sha256_file(output_path),
            )
        )
    return rendered
