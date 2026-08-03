from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import DatasetConfig
from ..schema import EpochRecord
from .preprocess import Annotation, PreprocessConfig, extract_epochs, preprocess_continuous


@dataclass(frozen=True, slots=True)
class Recording:
    cohort: str
    subject_id: str
    recording_id: str
    psg_path: Path
    annotation_path: Path


def normalize_stage(value: object) -> str | None:
    text = str(value).strip().upper()
    aliases = {
        "0": "W",
        "W": "W",
        "SLEEP-S0": "W",
        "1": "N1",
        "N1": "N1",
        "SLEEP-S1": "N1",
        "2": "N2",
        "N2": "N2",
        "SLEEP-S2": "N2",
        "3": "N3",
        "4": "N3",
        "N3": "N3",
        "N4": "N3",
        "SLEEP-S3": "N3",
        "SLEEP-S4": "N3",
        "5": "REM",
        "R": "REM",
        "REM": "REM",
        "SLEEP-REM": "REM",
    }
    return aliases.get(text)


def _subject_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.name), path.name) if path.name.isdigit() else (10**12, path.name)


def discover_recordings(config: DatasetConfig) -> list[Recording]:
    values = config.dataset
    cohort = str(values["name"])
    data_format = str(values["format"])

    if data_format == "shhs_profusion_xml":
        edf_root = Path(str(values["edf_root"]))
        annotation_root = Path(str(values["annotation_root"]))
        edfs = {
            path.stem: path for path in edf_root.glob(str(values.get("edf_glob", "*.edf")))
        }
        annotations = {}
        for path in annotation_root.glob(str(values.get("annotation_glob", "*.xml"))):
            key = path.stem.removesuffix("-profusion")
            annotations[key] = path
        return [
            Recording(cohort, key, key, edfs[key], annotations[key])
            for key in sorted(edfs.keys() & annotations.keys())
        ]

    if data_format in {"isruc_stage_txt", "isruc_xlsx"}:
        root = Path(str(values["raw_root"]))
        recordings = []
        for folder in sorted(root.glob(str(values.get("subject_glob", "*"))), key=_subject_sort_key):
            if not folder.is_dir():
                continue
            edfs = sorted(folder.glob(str(values.get("edf_glob", "*.edf"))))
            annotations = sorted(folder.glob(str(values.get("annotation_glob", "*_1.txt"))))
            if edfs and annotations:
                recordings.append(
                    Recording(cohort, folder.name, folder.name, edfs[0], annotations[0])
                )
        return recordings

    if data_format == "dcsm_ids":
        root_text = str(values.get("raw_root", ""))
        if not root_text:
            raise ValueError("DCSM raw_root is unresolved; update configs/datasets/dcsm.toml")
        root = Path(root_text)
        recordings = []
        for edf in sorted(root.glob(str(values.get("edf_glob", "**/*.edf")))):
            candidates = [
                path
                for path in sorted(edf.parent.glob(str(values.get("annotation_glob", "*.ids"))))
                if "processed" not in path.name.lower()
            ]
            if candidates:
                subject = edf.parent.name
                recordings.append(Recording(cohort, subject, edf.stem, edf, candidates[0]))
        return recordings

    raise ValueError(f"unsupported dataset format: {data_format}")


def load_annotations(recording: Recording, config: DatasetConfig) -> list[Annotation]:
    data_format = str(config.dataset["format"])
    if data_format == "dcsm_ids":
        annotations = []
        with recording.annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3:
                    raise ValueError(
                        f"invalid DCSM annotation at {recording.annotation_path}:{line_number}"
                    )
                annotations.append(
                    Annotation(float(parts[0]), float(parts[1]), normalize_stage(parts[2]))
                )
        return annotations

    if data_format == "shhs_profusion_xml":
        root = ET.parse(recording.annotation_path).getroot()
        return [
            Annotation(index * 30.0, 30.0, normalize_stage(node.text or ""))
            for index, node in enumerate(root.iter("SleepStage"))
        ]

    if data_format == "isruc_stage_txt":
        labels = recording.annotation_path.read_text(encoding="utf-8").splitlines()
        return [
            Annotation(index * 30.0, 30.0, normalize_stage(label))
            for index, label in enumerate(labels)
            if label.strip()
        ]

    if data_format == "isruc_xlsx":
        import pandas as pd

        frame = pd.read_excel(recording.annotation_path)
        column = int(config.dataset.get("annotation_stage_column", 1))
        return [
            Annotation(index * 30.0, 30.0, normalize_stage(row.iloc[column]))
            for index, (_, row) in enumerate(frame.iterrows())
        ]

    raise ValueError(f"unsupported annotation format: {data_format}")


def prepare_recording(
    recording: Recording,
    config: DatasetConfig,
    output_root: str | Path,
) -> list[EpochRecord]:
    try:
        import mne
    except ImportError as error:
        raise RuntimeError("MNE is required for raw EDF preparation") from error

    channels = [
        str(config.dataset["eeg_channel"]),
        str(config.dataset["eog_channel"]),
        str(config.dataset["emg_channel"]),
    ]
    raw = mne.io.read_raw_edf(recording.psg_path, preload=True, verbose="ERROR")
    missing = [channel for channel in channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{recording.recording_id} is missing channels {missing}")
    raw.pick(channels)
    raw.reorder_channels(channels)
    source_sfreq = float(raw.info["sfreq"])
    signal_volts = raw.get_data()
    preprocess_config = PreprocessConfig.from_dict(
        config.preprocessing,
        powerline_hz=float(config.dataset["powerline_hz"]),
    )
    signal_uv = preprocess_continuous(signal_volts, source_sfreq, preprocess_config)
    epochs = extract_epochs(signal_uv, load_annotations(recording, config), preprocess_config)
    if not epochs:
        raise ValueError(f"no valid epochs were produced for {recording.recording_id}")

    artifact_path = (
        Path(output_root) / recording.cohort.lower() / f"{recording.recording_id}.npz"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        signals=np.stack([epoch.signals_uv for epoch in epochs]),
        labels=np.asarray([epoch.label for epoch in epochs]),
        onsets_sec=np.asarray([epoch.onset_sec for epoch in epochs], dtype=np.float64),
        source_epoch_indices=np.asarray(
            [epoch.source_epoch_index for epoch in epochs], dtype=np.int64
        ),
        channels=np.asarray(["EEG", "EOG", "EMG"]),
        unit=np.asarray("uV"),
        target_sfreq=np.asarray(preprocess_config.target_sfreq),
    )

    records = []
    for artifact_index, epoch in enumerate(epochs):
        sample_id = (
            f"{recording.cohort}:{recording.subject_id}:{epoch.source_epoch_index:06d}"
        )
        records.append(
            EpochRecord(
                sample_id=sample_id,
                cohort=recording.cohort,
                subject_id=recording.subject_id,
                recording_id=recording.recording_id,
                epoch_index=epoch.source_epoch_index,
                onset_sec=epoch.onset_sec,
                label=epoch.label,
                source_psg_path=str(recording.psg_path.resolve()),
                source_annotation_path=str(recording.annotation_path.resolve()),
                source_sfreq=source_sfreq,
                artifact_path=str(artifact_path.resolve()),
                artifact_index=artifact_index,
            )
        )
    return records


def prepare_dataset(
    config: DatasetConfig,
    output_root: str | Path,
    *,
    limit: int | None = None,
) -> tuple[list[EpochRecord], list[dict[str, str]]]:
    recordings = discover_recordings(config)
    if limit is not None:
        recordings = recordings[:limit]
    records: list[EpochRecord] = []
    failures: list[dict[str, str]] = []
    for recording in recordings:
        try:
            records.extend(prepare_recording(recording, config, output_root))
        except Exception as error:
            failures.append(
                {
                    "recording_id": recording.recording_id,
                    "psg_path": str(recording.psg_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return records, failures
