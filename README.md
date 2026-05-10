# CodeAlpha — Handwritten Character Recognition

CNN-based handwritten **digit** recognition on MNIST using **PyTorch** (aligned with the internship brief; EMNIST letters can use the same architecture with more output classes).

## Requirements

- Python 3.10+ (tested through Python 3.14; installs CPU wheels from PyPI)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 12
```

Downloads MNIST into `.torch_data/` on first run. Saves `artifacts/mnist_cnn.pt` (state dict) and prints test accuracy.

## Streamlit UI

```bash
streamlit run app.py
```

Upload a digit image (dark strokes on light background work best; the app auto-inverts light-on-dark scans) or classify a MNIST test image by index.

## GitHub

Create a repository named `CodeAlpha_HandwrittenRecognition`, then:

```bash
git remote add origin https://github.com/<your-user>/CodeAlpha_HandwrittenRecognition.git
git branch -M main
git push -u origin main
```
