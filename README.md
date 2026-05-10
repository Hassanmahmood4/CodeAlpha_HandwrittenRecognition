# CodeAlpha — Handwritten Character Recognition

Enterprise-ready MNIST CNN built with **TensorFlow/Keras**, packaged Streamlit UX (upload → preview → logits), and Docker automation for environments where TensorFlow wheels are unavailable locally.

## Architecture

- Convolutional feature extractor + dense classifier with dropout regularization.
- `Rescaling` layer keeps training/inference normalization identical.
- Metrics persisted to `artifacts/training_metrics.json`; weights saved as `artifacts/mnist_cnn.keras`.

## Local prerequisites

TensorFlow publishes wheels for **Python 3.9–3.12**. Newer interpreters (for example 3.14 preview builds) may lack wheels — use Docker or a 3.12 virtual environment.

### Setup (Python 3.11/3.12 recommended)

```bash
cd CodeAlpha_HandwrittenRecognition
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/train_model.py
streamlit run app.py
```

### Docker (works everywhere Docker runs)

```bash
cd CodeAlpha_HandwrittenRecognition
docker build -t codealpha-mnist .
docker run --rm -p 8501:8501 codealpha-mnist
```

Visit `http://localhost:8501`. The image trains MNIST during `docker build`, so the first build takes longer.

## Streamlit workflow

1. Upload a digit-centric PNG/JPEG/WebP.
2. Inspect the preview pane (auto inversion for light backgrounds).
3. Tap **Run CNN inference** for softmax probabilities + chart/table.

## Modular layout

```
src/
  config.py        # constants / artifact paths
  model.py         # CNN definition
  preprocessing.py # PIL → tensor utility
scripts/
  train_model.py   # CLI trainer
app.py             # Streamlit entrypoint
```

## GitHub

Publish as `CodeAlpha_HandwrittenRecognition`:

```bash
git remote add origin https://github.com/<your-user>/CodeAlpha_HandwrittenRecognition.git
git branch -M main
git push -u origin main
```

## Operational notes

- For GPU hosts, swap the base image or install `tensorflow` GPU builds per NVIDIA guidance.
- `.streamlit/config.toml` tweaks palette + typography for hosted demos.
