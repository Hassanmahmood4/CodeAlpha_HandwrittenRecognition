#!/usr/bin/env python3
"""
Production-ready handwritten digit desk — TensorFlow-first with resilient PyTorch fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

from src.config import METRICS_PATH  # noqa: E402
from src.inference import load_digit_predictor  # noqa: E402
from src.preprocessing import pil_to_model_tensor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            div[data-testid="stAppViewContainer"] {
                background: radial-gradient(circle at 15% 20%, #431407 0%, #020617 65%, #020617 100%);
            }
            .block-container { padding-top: 1.65rem; padding-bottom: 3rem; max-width: 1050px; }
            .glass-panel {
                border: 1px solid rgba(251,146,60,.35);
                border-radius: 18px;
                padding: 1.25rem 1.35rem;
                background: rgba(2,6,23,.82);
                box-shadow: 0 28px 55px rgba(2,6,23,.65);
                backdrop-filter: blur(16px);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_metrics() -> dict | None:
    if not METRICS_PATH.is_file():
        return None
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def tensor_preview(tensor: np.ndarray) -> np.ndarray:
    """Normalized grayscale preview for QA visualization."""
    slice_ = tensor.reshape(28, 28)
    lo, hi = slice_.min(), slice_.max()
    if hi - lo < 1e-6:
        return np.zeros_like(slice_)
    return (slice_ - lo) / (hi - lo)


def main() -> None:
    st.set_page_config(page_title="Digit Cognition Studio", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="glass-panel">
            <p style="letter-spacing:.22em;text-transform:uppercase;color:#fdba74;font-size:.78rem;margin-bottom:.35rem;">
                CNN inference theater</p>
            <h1 style="margin:0;color:#fff7ed;font-size:2rem;">Handwritten digit intelligence studio</h1>
            <p style="margin-top:.75rem;color:#fed7aa;line-height:1.55rem;">
                TensorFlow/Keras whenever wheels exist — seamless PyTorch fallback keeps internships demo-ready on bleeding-edge Python builds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor, backend_hint = load_digit_predictor()
    metrics = load_metrics()

    if predictor is None:
        st.error("Unable to load inference artifacts.")
        if backend_hint == "missing_checkpoint":
            st.code(f"cd {PROJECT_ROOT} && python scripts/train_model.py && python scripts/smoke_test.py", language="bash")
        else:
            st.warning("Weights detected but deserialization failed — reinstall dependencies or rebuild Docker image.")
        return

    badge = "TensorFlow · Keras native" if backend_hint == "tensorflow" else "PyTorch · parity fallback"
    st.sidebar.markdown(f"**Runtime lane**\n\n`{badge}`")
    if metrics:
        st.sidebar.metric("Hold-out accuracy", f"{metrics['test_accuracy']:.2%}")
        if metrics.get("test_loss"):
            st.sidebar.metric("Hold-out loss", f"{metrics['test_loss']:.4f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Capture playbook**\n"
        "- Center strokes inside the crop.\n"
        "- Prefer ≥600×600 uploads.\n"
        "- Light backgrounds invert automatically."
    )

    uploaded = st.file_uploader("Upload handwritten digit (PNG / JPEG / WEBP)", type=["png", "jpg", "jpeg", "webp"])

    preview_col, chart_col = st.columns((1.05, 1))

    tensor = None
    pil_image = None
    if uploaded is not None:
        from PIL import Image

        pil_image = Image.open(uploaded)

    if pil_image is None:
        preview_col.info("Awaiting upload — rendered preview & tensor QA tiles unlock automatically.")
    else:
        preview_col.image(pil_image, caption="Original capture", use_container_width=True)
        tensor = pil_to_model_tensor(pil_image)
        qa_preview = tensor_preview(tensor)
        preview_col.caption("Normalized 28×28 MNIST tensor preview")
        preview_col.image(qa_preview, caption="Preprocessed tensor (scaled for display)", width=160)

    infer = st.button("Run CNN inference", type="primary", disabled=tensor is None)
    if infer and tensor is None:
        st.warning("Upload an image before invoking inference.")
    elif infer and tensor is not None:
        with st.spinner("Executing CNN forward pass · stabilizing softmax probabilities..."):
            probs_batch = predictor(tensor)
        probs = probs_batch[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100.0)

        st.success(f"Predicted digit · **{digit}**")
        metrics_row = st.columns(3)
        metrics_row[0].metric("Top-class confidence", f"{confidence:.2f}%")
        metrics_row[1].metric("Runner-up mass", f"{float(np.partition(probs, -2)[-2]) * 100:.2f}%")
        metrics_row[2].metric("Entropy proxy", f"{float(-np.sum(probs * np.log(probs + 1e-9))):.3f}")

        df_chart = pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)])
        chart_col.markdown("#### Probability simplex")
        chart_col.bar_chart(df_chart)

        with st.expander("Raw softmax vector"):
            st.dataframe(df_chart.style.format("{:.5f}"))

    if pil_image is None:
        chart_col.info("Probability dashboard activates post-upload.")


if __name__ == "__main__":
    main()
