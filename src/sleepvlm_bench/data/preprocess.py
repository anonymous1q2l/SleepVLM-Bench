from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from ..constants import STAGES


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    target_sfreq: float = 100.0
    epoch_seconds: int = 30
    filter_order: int = 4
    anti_alias_hz: float = 48.0
    eeg_band_hz: tuple[float, float] = (0.3, 35.0)
    eog_band_hz: tuple[float, float] = (0.1, 10.0)
    emg_band_hz: tuple[float, float] = (10.0, 45.0)
    powerline_hz: float | None = None
    notch_quality_factor: float = 30.0
    normalize: bool = False

    @classmethod
    def from_dict(
        cls, values: dict[str, object], *, powerline_hz: float | None = None
    ) -> "PreprocessConfig":
        return cls(
            target_sfreq=float(values.get("target_sfreq", 100.0)),
            epoch_seconds=int(values.get("epoch_seconds", 30)),
            filter_order=int(values.get("filter_order", 4)),
            anti_alias_hz=float(values.get("anti_alias_hz", 48.0)),
            eeg_band_hz=tuple(float(v) for v in values.get("eeg_band_hz", (0.3, 35.0))),
            eog_band_hz=tuple(float(v) for v in values.get("eog_band_hz", (0.1, 10.0))),
            emg_band_hz=tuple(float(v) for v in values.get("emg_band_hz", (10.0, 45.0))),
            powerline_hz=powerline_hz,
            normalize=bool(values.get("normalize", False)),
        )

    def validate(self, source_sfreq: float) -> None:
        if source_sfreq <= 0 or self.target_sfreq <= 0:
            raise ValueError("sampling rates must be positive")
        if self.epoch_seconds <= 0 or self.filter_order <= 0:
            raise ValueError("epoch length and filter order must be positive")
        if self.normalize:
            raise ValueError("the paper protocol preserves amplitude and forbids normalization")
        nyquist = source_sfreq / 2.0
        for name, (low, high) in zip(
            ("EEG", "EOG", "EMG"),
            (self.eeg_band_hz, self.eog_band_hz, self.emg_band_hz),
            strict=True,
        ):
            if not 0 < low < high < nyquist:
                raise ValueError(
                    f"{name} band {(low, high)} is invalid for source sfreq {source_sfreq}"
                )


@dataclass(frozen=True, slots=True)
class Annotation:
    onset_sec: float
    duration_sec: float
    label: str | None


@dataclass(frozen=True, slots=True)
class PreparedEpoch:
    source_epoch_index: int
    onset_sec: float
    label: str
    signals_uv: np.ndarray


def _notch(data: np.ndarray, sfreq: float, frequency: float | None, quality: float) -> np.ndarray:
    if frequency is None or frequency >= sfreq / 2.0:
        return data
    b, a = iirnotch(frequency, quality, fs=sfreq)
    return filtfilt(b, a, data, axis=-1)


def _bandpass(
    data: np.ndarray, sfreq: float, band: tuple[float, float], order: int
) -> np.ndarray:
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _anti_alias(data: np.ndarray, sfreq: float, cutoff: float, order: int) -> np.ndarray:
    if cutoff >= sfreq / 2.0:
        return data
    sos = butter(order, cutoff, btype="lowpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def preprocess_continuous(
    signals_volts: np.ndarray, source_sfreq: float, config: PreprocessConfig
) -> np.ndarray:
    """Apply the manuscript filter order to continuous EEG/EOG/EMG data.

    Input shape is `(3, samples)` in volts. Output is `(3, samples)` at the target
    sampling rate in microvolts. Filtering is performed before epoch extraction.
    """

    values = np.asarray(signals_volts, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"expected signals with shape (3, samples), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("signals contain non-finite values")
    config.validate(source_sfreq)

    filtered = _notch(values, source_sfreq, config.powerline_hz, config.notch_quality_factor)
    bands = (config.eeg_band_hz, config.eog_band_hz, config.emg_band_hz)
    filtered = np.stack(
        [
            _bandpass(filtered[index], source_sfreq, band, config.filter_order)
            for index, band in enumerate(bands)
        ]
    )
    if source_sfreq > config.target_sfreq:
        filtered = _anti_alias(
            filtered, source_sfreq, config.anti_alias_hz, config.filter_order
        )

    ratio = Fraction(config.target_sfreq / source_sfreq).limit_denominator(10_000)
    if ratio.numerator != ratio.denominator:
        filtered = resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1)
    return (filtered * 1_000_000.0).astype(np.float32)


def extract_epochs(
    signals_uv: np.ndarray,
    annotations: Iterable[Annotation],
    config: PreprocessConfig,
) -> list[PreparedEpoch]:
    samples_per_epoch = int(round(config.target_sfreq * config.epoch_seconds))
    epochs: list[PreparedEpoch] = []
    for annotation in annotations:
        complete_epochs = int(annotation.duration_sec // config.epoch_seconds)
        for offset in range(complete_epochs):
            onset = annotation.onset_sec + offset * config.epoch_seconds
            source_epoch_index = int(round(onset / config.epoch_seconds))
            if annotation.label not in STAGES:
                continue
            start = int(round(onset * config.target_sfreq))
            stop = start + samples_per_epoch
            if start < 0 or stop > signals_uv.shape[-1]:
                continue
            epoch = signals_uv[:, start:stop]
            if epoch.shape != (3, samples_per_epoch):
                continue
            epochs.append(
                PreparedEpoch(
                    source_epoch_index=source_epoch_index,
                    onset_sec=onset,
                    label=annotation.label,
                    signals_uv=epoch,
                )
            )
    return epochs

