#!/usr/bin/env python3
"""
Streamlit digit intelligence desk powered by TensorFlow/Keras CNN checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    keras = None  # type: ignore
    TF_IMPORT_ERROR = exc
else:
    TF_IMPORT_ERROR = None

from src.config import METRICS_PATH, MODEL_PATH
from src.preprocessing import pil_to_model_tensor

PROJECT_ROOT = Path(__file__).resolve().parent


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.25rem; max-width: 960px; }
            div[data-testid="stVerticalBlock"] > div { gap: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model_safe():  # pragma: no cover - tensorflow runtime specific
    if keras is None:
        return None
    if not MODEL_PATH.is_file():
        return None
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def load_metrics() -> dict | None:
    if not METRICS_PATH.is_file():
        return None
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    st.set_page_config(page_title="Digit Cognition Studio", layout="wide")
    inject_styles()

    st.title("Handwritten digit cognition studio")
    st.caption("TensorFlow/Keras CNN • MNIST-trained weights • upload-first UX")

    if TF_IMPORT_ERROR is not None:
        st.error("TensorFlow could not be imported on this interpreter.")
        st.code(str(TF_IMPORT_ERROR))
        st.warning(
            "TensorFlow currently ships wheels for **Python 3.9–3.12**. "
            "Use `docker compose up` (recommended) or install via pyenv/conda per README."
        )
        return

    model = load_model_safe()
    metrics = load_metrics()

    if model is None:
        st.error("No trained checkpoint detected.")
        st.code(f"cd {PROJECT_ROOT} && python scripts/train_model.py", language="bash")
        st.info(
            "Dockerfile bundles Python 3.12 + TensorFlow so training/inference works without local TF wheels."
        )
        return

    if metrics:
        st.sidebar.metric("Hold-out accuracy", f"{metrics['test_accuracy']:.2%}")
        st.sidebar.metric("Hold-out loss", f"{metrics['test_loss']:.4f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tips**\n"
        "- Center digits inside the frame.\n"
        "- High-contrast strokes produce sharper logits.\n"
        "- Light backgrounds auto-invert toward MNIST polarity."
    )

    uploaded = st.file_uploader("Upload handwritten digit (PNG/JPEG/WebP)", type=["png", "jpg", "jpeg", "webp"])

    col_preview, col_chart = st.columns((1, 1))

    tensor = None
    preview_image = None

    if uploaded is not None:
        from PIL import Image

        image = Image.open(uploaded)
        preview_image = image
        tensor = pil_to_model_tensor(image)

    if preview_image is None:
        col_preview.info("Awaiting upload — preview will render here.")
    else:
        col_preview.image(preview_image, caption="Original upload", use_container_width=True)

    infer = st.button("Run CNN inference", type="primary")
    if infer and tensor is None:
        st.warning("Upload an image before running inference.")
    elif infer and tensor is not None:
        preds = model.predict(tensor, verbose=0)
        probs = preds[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100.0)

        st.success(f"Predicted digit: **{digit}**")
        st.metric("Top-class confidence", f"{confidence:.2f}%")

        df_chart = pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)])
        col_chart.bar_chart(df_chart)

        with st.expander("Probability table"):
            st.dataframe(df_chart.style.format("{:.4f}"))


if __name__ == "__main__":
    main()
