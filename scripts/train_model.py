#!/usr/bin/env python3
"""
Train MNIST CNN with TensorFlow/Keras and persist weights + metrics.

Usage:

    python scripts/train_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensorflow import keras  # noqa: E402

from src.config import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATH  # noqa: E402
from src.model import build_cnn  # noqa: E402


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = build_cnn()
    early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=128,
        callbacks=[early_stop],
        verbose=2,
    )

    metrics_results = model.evaluate(x_test, y_test, verbose=0)
    payload = {
        "test_loss": float(metrics_results[0]),
        "test_accuracy": float(metrics_results[1]),
        "epochs_trained": len(history.history["loss"]),
    }
    model.save(MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
