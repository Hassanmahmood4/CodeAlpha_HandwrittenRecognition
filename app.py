#!/usr/bin/env python3
"""Streamlit UI for MNIST digit classification with the trained CNN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import datasets, transforms

from mnist_model import MnistCNN

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TORCH_DATA = ROOT / ".torch_data"


def load_trained_model():
    path = ARTIFACTS / "mnist_cnn.pt"
    if not path.is_file():
        return None, None
    map_loc = torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_loc)
    model = MnistCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, path


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    img = Image.fromarray(arr.astype(np.uint8))
    tfm = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return tfm(img).unsqueeze(0)


@st.cache_resource
def mnist_test_dataset():
    TORCH_DATA.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return datasets.MNIST(root=str(TORCH_DATA), train=False, download=True, transform=tfm)


def main():
    st.set_page_config(page_title="Handwritten digits", layout="centered")
    st.title("Handwritten digit recognition")
    st.caption("Upload a digit image (roughly centered) or sample from MNIST test set. Train first: `python train.py`.")

    model, weights_path = load_trained_model()
    if model is None:
        st.warning(f"No weights at `{ARTIFACTS / 'mnist_cnn.pt'}`. Run training, then refresh.")
        st.code("python train.py --epochs 12", language="bash")
        return

    st.caption(f"Loaded weights: `{weights_path.name}`")

    mode = st.radio("Input", ("Upload image", "MNIST test sample"), horizontal=True)

    x = None
    if mode == "Upload image":
        up = st.file_uploader("Image (PNG/JPEG)", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            img = Image.open(up)
            st.image(img, caption="Original", width=200)
            x = preprocess_pil(img)
    else:
        ds = mnist_test_dataset()
        idx = st.slider("Test image index", 0, len(ds) - 1, 0)
        tensor, true_label = ds[idx]
        viz = tensor.squeeze().numpy() * 0.3081 + 0.1307
        viz = np.clip(viz, 0.0, 1.0)
        st.image(viz, caption=f"MNIST test #{idx} (label {true_label})", width=160)
        x = tensor.unsqueeze(0)

    if x is not None and st.button("Classify", type="primary"):
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy().ravel()
        pred = int(probs.argmax())
        st.success(f"Predicted digit: **{pred}**")
        st.bar_chart(
            pd.DataFrame({"P(class)": probs.tolist()}, index=[str(i) for i in range(10)])
        )


if __name__ == "__main__":
    main()
