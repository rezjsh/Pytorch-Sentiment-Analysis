# Pytorch-Sentiment-Analysis

A modular, end-to-end sentiment analysis project supporting both classic Machine Learning (TF-IDF + linear models) and Deep Learning (LSTM/CNN/BERT/SBERT) approaches, with a beautiful Streamlit app for interactive inference and reporting.

This repository provides:
- Reproducible data preprocessing and training pipelines
- Configurable ML and DL model choices via YAML
- Baseline ML classifier (Logistic Regression / Linear SVM)
- Deep Learning model definitions (LSTM, CNN, BiLSTM+Attention, BERT, SBERT)
- Modular Streamlit app for single-text and CSV-batch predictions, with cached checkpoints and feature-importance visuals


## Contents
- Features
- Project Structure
- Installation
- Quickstart
- Streamlit App
- Training & Evaluation Pipelines (CLI)
- Configuration (configs/config.yaml, configs/params.yaml)
- Checkpoints, Reports & Artifacts
- Testing
- Docker
- Troubleshooting
- License


## Features
- Data pipeline using Hugging Face datasets (default: imdb)
- ML baseline via TF-IDF + Logistic Regression or Linear SVM
- DL models: LSTM, CNN, BiLSTM with Attention, BERT, SBERT
- Modular components and pipelines under `src/sentiment_analysis`
- Clean logging
- Streamlit app with:
  - Single-text prediction
  - Batch predictions from CSV
  - Optional metrics (accuracy, classification report, confusion matrix) if label column is provided
  - Feature-importance bar charts for ML baseline


## Project Structure
```
.
├─ app/                         # Modular Streamlit app package
│  ├─ __init__.py
│  ├─ config.py                 # Paths and app configuration helpers
│  ├─ services.py               # Baseline model loading/training and prediction
│  └─ ui.py                     # Streamlit UI layout and tabs
├─ app_streamlit.py             # Thin entrypoint for Streamlit, calls app.ui.run_app
├─ configs/
│  ├─ config.yaml               # High-level project configuration (mode, data, model, trainer)
│  └─ params.yaml               # Model-specific hyperparameters and ML baseline params
├─ src/sentiment_analysis/
│  ├─ components/               # Reusable building blocks (preprocessing, ML baseline, EDA, trainer, ...)
│  ├─ models/                   # DL model implementations
│  ├─ pipeline/                 # Orchestrated pipeline stages
│  ├─ utils/                    # Logging, helpers
│  └─ ...
├─ checkpoints/                 # Saved models/checkpoints (created at runtime)
├─ reports/                     # Reports/metrics/plots (created at runtime)
├─ main.py                      # CLI entry to run pipelines
├─ requirements.txt
├─ Dockerfile
└─ README.md
```


## Installation
Requirements:
- Python 3.9+ (recommended 3.10/3.11)
- Pip >= 21
- Git

Steps (Windows PowerShell / Command Prompt):
```
# 1) (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# If streamlit not installed by requirements.txt
pip install streamlit
```

If you plan to run Deep Learning models on GPU, install a CUDA-enabled PyTorch build. Refer to https://pytorch.org/get-started/locally/


## Quickstart
Run the Streamlit app (recommended for first-time exploration):
```
streamlit run app_streamlit.py
```
- Choose LOGREG or SVM on the sidebar
- Keep the dataset as imdb (or change to another Hugging Face dataset with `text`/`label` columns)
- On first run, if no checkpoint exists, the app trains a quick baseline and caches it under `checkpoints/`, and writes a report to `reports/`


## Streamlit App
The Streamlit UI is modularized under `app/` and the entrypoint is `app_streamlit.py`.

Tabs:
- Single Text
  - Enter text and get Positive/Negative
- Batch (CSV)
  - Upload a CSV with a required `text` column and optional `label` column (0/1)
  - Runs predictions and, if `label` exists, reports Accuracy, classification report, and a confusion matrix
  - Download predictions as CSV
- Model Report
  - Shows last evaluation accuracy and bar charts for top positive/negative words (from ML baseline report JSON)

Caching & Artifacts:
- Trained ML baselines are saved to `checkpoints/{model}_baseline.joblib`
- Report files are saved to `reports/{model}_report.json`

Modular App Code:
- `app/config.py` centralizes runtime paths
- `app/services.py` encapsulates training/loading of the baseline model and prediction
- `app/ui.py` renders UI and tabs and wires everything together


## Training & Evaluation Pipelines (CLI)
The project includes pipeline stages to run end-to-end processing, training, and evaluation.

Entry point: `main.py`
- The pipeline mode (ML vs DL) is controlled by `configs/config.yaml`.
- It orchestrates the following (depending on mode):
  - Stage 01: Data Preprocessing (splits/tokenization)
  - Stage 02: Dataset & DataLoader (DL) OR ML Baseline training (ML)
  - Stage 03: EDA (DL)
  - Stage 04: Model creation (DL)
  - Stage 05: Callbacks creation (DL)
  - Stage 06: Model training (DL)
  - Stage 07: Evaluation (DL)

Run:
```
python main.py
```

Note: Ensure `configs/config.yaml` is set to your desired mode and model type.


## Configuration
Configuration is split into two files:

1) `configs/config.yaml` (project-wide settings)
- `mode`: 'ML' or 'DL'
- `data`:
  - `dataset_name`: Hugging Face dataset ID (default: imdb)
  - `max_length`: tokenizer max length (DL)
  - `batch_size`: global batch size (DL)
  - `test_split_ratio`: test split portion
  - `seed`: random seed
  - `tokenizer_name`: e.g., bert-base-uncased
- `ml_baseline`:
  - `classifier_name`: LOGREG or SVM
  - `model_path`: where to save the baseline model
  - `report_path`: where to save metrics
- `model`:
  - `model_type`: one of LSTM, CNN, LSTMATTENTION, BERT, SBERT
- `callbacks`, `trainer`, `evaluation`: paths and settings for checkpoints and reports

2) `configs/params.yaml` (hyperparameters)
- `ml_baseline`:
  - `max_features`: TF-IDF vocabulary size
  - `ngram_range`: e.g. [1,2] for unigrams+bigrams
  - `C`: regularization
  - `max_iter`: training iterations for linear models
- `model_options`: per-model hyperparameters (LSTM, CNN, BERT, SBERT, etc.)
- `callbacks`: early stopping, LR scheduler, gradient clipping
- `trainer`: learning rate and related training parameters

Tip: Keep file paths consistent with your environment and ensure directories exist (the project usually creates them as needed).


## Checkpoints, Reports & Artifacts
- ML Baseline
  - Model: `checkpoints/{model}_baseline.joblib` (app)
  - Report: `reports/{model}_report.json` (app)
  - Optional: configuration `ml_baseline.model_path` / `ml_baseline.report_path` (pipelines)
- DL Training
  - Checkpoints: `checkpoints/`
  - Trainer reports: `reports/model_reports/`
  - Evaluation: `reports/evaluation_reports/`
- EDA
  - Reports: `reports/eda_reports/`


## Testing
A `tests/` directory is included to add unit and integration tests.

Recommended:
```
pip install pytest
pytest -q
```


## Docker
A basic Dockerfile is included. Example usage:
```
# Build
docker build -t pytorch-sentiment-app .

# Run Streamlit app on port 8501
docker run -p 8501:8501 pytorch-sentiment-app \
  streamlit run app_streamlit.py --server.address=0.0.0.0 --server.port=8501
```

If running DL training with GPU, build against a CUDA base image and use `--gpus all` on supported hosts.


## Troubleshooting
- Hugging Face dataset download issues
  - Ensure internet connectivity; datasets are cached automatically
  - Try a different dataset or split
- Streamlit cannot find modules
  - Ensure you run from the project root so `src/` is on the path (setup.py may add packages).
- PreprocessingConfig arguments
  - The app sets `batch_size=32` and `tokenizer_name="bert-base-uncased"` to satisfy the dataclass. If you change the Preprocessing class to require different args, update `app/services.py` accordingly.
- GPU/CPU
  - If CUDA is available, DL training may use it. Ensure correct PyTorch build is installed.


## License
This project is licensed under the terms of the MIT License. See the LICENSE file for details.
