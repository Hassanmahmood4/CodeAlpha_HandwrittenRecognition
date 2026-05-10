"""Load the best available inference backend (TensorFlow/Keras or PyTorch)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _build_tf_predictor():
    try:
        from tensorflow import keras  # noqa: WPS433
    except ImportError:
        return None

    from .config import MODEL_PATH  # noqa: WPS433

    if not MODEL_PATH.is_file():
        return None
    model = keras.models.load_model(MODEL_PATH)

    def predict(batch_nhwc: np.ndarray) -> np.ndarray:
        preds = model.predict(batch_nhwc, verbose=0)
        return np.asarray(preds, dtype=np.float64)

    return predict, "tensorflow"


def _build_torch_predictor():
    import torch  # noqa: WPS433

    from .config import TORCH_CKPT  # noqa: WPS433
    from .torch_model import DigitCNN  # noqa: WPS433

    if not TORCH_CKPT.is_file():
        return None
    try:
        raw = torch.load(TORCH_CKPT, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        raw = torch.load(TORCH_CKPT, map_location=torch.device("cpu"))
    net = DigitCNN()
    net.load_state_dict(raw["state_dict"])
    net.eval()

    def predict(batch_nhwc: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(batch_nhwc.astype(np.float32)).permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = net(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict, "pytorch"


def load_digit_predictor():
    """Return ``(predict_fn, backend_name)`` or ``(None, reason)``."""
    tf_bundle = _build_tf_predictor()
    if tf_bundle:
        return tf_bundle

    torch_bundle = _build_torch_predictor()
    if torch_bundle:
        return torch_bundle

    from .config import MODEL_PATH, TORCH_CKPT  # noqa: WPS433

    if not MODEL_PATH.is_file() and not TORCH_CKPT.is_file():
        return None, "missing_checkpoint"
    return None, "load_failure"
