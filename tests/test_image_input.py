import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sleepvlm_bench.inference.openai_compatible import image_data_url


class ImageInputTests(unittest.TestCase):
    def test_api_image_is_resized_and_letterboxed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (400, 200), "black").save(source)
            data_url = image_data_url(source, 224)
            prefix, encoded = data_url.split(",", maxsplit=1)
            self.assertEqual(prefix, "data:image/png;base64")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as resized:
                self.assertEqual(resized.size, (224, 224))

    def test_invalid_image_size_is_rejected(self):
        with self.assertRaises(ValueError):
            image_data_url(Path("missing.png"), 0)


if __name__ == "__main__":
    unittest.main()

