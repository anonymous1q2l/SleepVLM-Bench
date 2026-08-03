from __future__ import annotations

import base64
import io
import os
from pathlib import Path


def image_data_url(image_path: Path, image_size: int) -> str:
    from PIL import Image, ImageOps

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    with Image.open(image_path) as source:
        image = ImageOps.pad(
            source.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.BICUBIC,
            color="white",
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str,
        base_url: str | None = None,
        model_revision: str = "",
        image_size: int = 336,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install sleepvlm-bench[api] to use API inference") from error
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"required API credential is absent: {api_key_env}")
        self.model_id = model_id
        self.model_revision = model_revision
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.decoding_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "image_size": image_size,
        }
        self.backend_config = {
            "backend": "openai_compatible",
            "base_url": base_url or "https://api.openai.com/v1",
            "api_key_env": api_key_env,
        }
        client_options = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url
        self._client = OpenAI(**client_options)

    def predict(self, image_path: Path, prompt: str) -> str:
        data_url = image_data_url(image_path, self.image_size)
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stream=False,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("model returned an empty response")
        return content
