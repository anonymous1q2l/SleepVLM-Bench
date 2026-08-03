from __future__ import annotations

from pathlib import Path
from typing import Protocol


class VisionLanguageBackend(Protocol):
    model_id: str
    model_revision: str
    decoding_config: dict[str, object]
    backend_config: dict[str, object]

    def predict(self, image_path: Path, prompt: str) -> str:
        """Return the unmodified text response from the model."""
