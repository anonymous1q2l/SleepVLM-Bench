from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .constants import STAGES


@dataclass(frozen=True, slots=True)
class EpochRecord:
    sample_id: str
    cohort: str
    subject_id: str
    recording_id: str
    epoch_index: int
    onset_sec: float
    label: str
    source_psg_path: str
    source_annotation_path: str
    source_sfreq: float
    artifact_path: str
    artifact_index: int
    split: str = ""
    image_path: str = ""
    image_sha256: str = ""

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.cohort or not self.subject_id or not self.recording_id:
            raise ValueError(f"identity fields are incomplete for {self.sample_id}")
        if self.label not in STAGES:
            raise ValueError(f"invalid label {self.label!r} for {self.sample_id}")
        if self.epoch_index < 0 or self.artifact_index < 0 or self.onset_sec < 0:
            raise ValueError(f"negative epoch metadata for {self.sample_id}")
        if self.source_sfreq <= 0:
            raise ValueError(f"invalid source sampling rate for {self.sample_id}")
        if self.split not in {"", "train", "validation", "test"}:
            raise ValueError(f"invalid split {self.split!r} for {self.sample_id}")
        if bool(self.image_path) != bool(self.image_sha256):
            raise ValueError(
                f"image_path and image_sha256 must be set together for {self.sample_id}"
            )

    def to_row(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "EpochRecord":
        record = cls(
            sample_id=row["sample_id"],
            cohort=row["cohort"],
            subject_id=row["subject_id"],
            recording_id=row["recording_id"],
            epoch_index=int(row["epoch_index"]),
            onset_sec=float(row["onset_sec"]),
            label=row["label"],
            source_psg_path=row["source_psg_path"],
            source_annotation_path=row["source_annotation_path"],
            source_sfreq=float(row["source_sfreq"]),
            artifact_path=row["artifact_path"],
            artifact_index=int(row["artifact_index"]),
            split=row.get("split", ""),
            image_path=row.get("image_path", ""),
            image_sha256=row.get("image_sha256", ""),
        )
        record.validate()
        return record


MANIFEST_COLUMNS = tuple(EpochRecord.__dataclass_fields__)
