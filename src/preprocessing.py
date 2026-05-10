"""Input sanitation for uploaded handwriting."""
from __future__ import annotations

import numpy as np
from PIL import Image


def pil_to_model_tensor(image: Image.Image, invert_light_background: bool = True) -> np.ndarray:
    """
    Convert arbitrary PIL uploads into a MNIST-shaped tensor (1,28,28,1) float32.

    Values remain in 0–255 space because the Keras model applies ``Rescaling`` internally.
    """
    gray = image.convert("L")
    gray = gray.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    if invert_light_background and arr.mean() > 127.0:
        arr = 255.0 - arr
    return np.expand_dims(np.expand_dims(arr, -1), 0)
