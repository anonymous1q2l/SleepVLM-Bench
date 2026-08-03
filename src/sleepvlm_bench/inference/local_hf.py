from __future__ import annotations

from pathlib import Path


class LocalTransformersBackend:
    """Generic Transformers backend for verified image-to-text model revisions.

    Model-specific smoke tests are required before a model is added to a paper run.
    Transformers architectures do not all share identical chat templates.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        image_size: int,
        max_new_tokens: int = 2048,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "install sleepvlm-bench[local-vlm] to use local VLM inference"
            ) from error
        if not model_revision:
            raise ValueError("local paper runs require an immutable model revision")
        self.model_id = model_id
        self.model_revision = model_revision
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        self.decoding_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "image_size": image_size,
        }
        self.backend_config = {
            "backend": "local_hf",
            "device_map": device_map,
        }
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=model_revision,
            torch_dtype="auto",
            device_map=device_map,
        )
        self._model.eval()

    def predict(self, image_path: Path, prompt: str) -> str:
        from PIL import Image, ImageOps

        with Image.open(image_path) as source:
            image = ImageOps.pad(
                source.convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.BICUBIC,
                color="white",
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=[formatted], images=[image], return_tensors="pt")
        inputs = {
            name: value.to(self._model.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        input_length = inputs["input_ids"].shape[-1]
        return self._processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True
        )[0].strip()
