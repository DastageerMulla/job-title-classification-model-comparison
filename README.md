# Job Title Categorization: RoBERTa vs JobBERT-v2

This deep-learning project categorizes free-text job titles into one of **64 job domains** and compares predictions from two fine-tuned transformer models:

- **Old model:** RoBERTa sequence classifier
- **New model:** JobBERT-v2, using an MPNet sequence-classification architecture

For each title, the pipeline saves the top three categories and confidence scores from both models. It can then analyze model agreement, confidence, category-distribution shift, and cases that may need human review.

> **Important:** this is currently a model-to-model comparison, not a ground-truth accuracy test. True accuracy cannot be calculated without human-verified labels.

## Features

- Batch processing for large CSV files
- Automatic CUDA GPU use when available, with CPU fallback
- Top-three predictions and confidence scores from both models
- Cohen's kappa, Jaccard similarity, and Jensen-Shannon distance
- CSV export of disagreements for manual review
- HTML comparison-interface template

## Project structure

```text
.
|-- models/
|   |-- old_model/                 # Fine-tuned RoBERTa model
|   `-- new_model/                 # Fine-tuned JobBERT-v2/MPNet model
|-- templates/
|   `-- job_mapper_compare.html    # Comparison UI template
|-- run.py                         # Batch inference for both models
|-- script.py                      # Model-comparison analysis
|-- searchString.csv               # Input titles (one per row)
|-- output2.csv                    # Predictions from run.py
`-- needs_human_review.csv         # Review cases from script.py
```

## Requirements

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- pandas, NumPy, scikit-learn, SciPy, and joblib

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install torch transformers pandas numpy scikit-learn scipy joblib
```

For CUDA support, use the installation command recommended for your system on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Model setup

The model folders must contain their Hugging Face configuration, tokenizer, weights, and label files. The new model also requires `label_encoder.pkl`.

Before running inference, change the machine-specific paths near the top of `run.py`. The models included in this project can be referenced with relative paths:

```python
OLD_MODEL_DIR = os.path.join(
    "models",
    "old_model",
    "job_domain_roberta_model-20260113T104916Z-3-001",
    "job_domain_roberta_model",
)
NEW_MODEL_DIR = os.path.join("models", "new_model", "jobbert_v2_finetuned")
```

The network paths currently saved in `run.py` will not work on another computer unchanged.

## Input format

Put job titles in the first column of `searchString.csv`. A header is not required:

```csv
acute care
pediatrics
mental health
software engineer
```

The input, output, and batch size can be changed in `run.py`:

```python
INPUT_CSV_PATH = "searchString.csv"
OUTPUT_CSV_PATH = "output2.csv"
BATCH_SIZE = 50
```

## Usage

### 1. Generate predictions

```bash
python run.py
```

This loads both models, runs batch inference, and writes `output2.csv`. The output contains the original title plus three category/confidence pairs from each model:

```text
id, search_term,
conf_old_1, cat_old_1, conf_old_2, cat_old_2, conf_old_3, cat_old_3,
conf_new_1, cat_new_1, conf_new_2, cat_new_2, conf_new_3, cat_new_3
```

Confidence values are stored as percentages on a `0-100` scale.

### 2. Analyze the comparison

```bash
python script.py
```

This prints a report and writes selected disagreements to `needs_human_review.csv`. The report covers:

- Top-one agreement and Cohen's kappa
- Mean top-three Jaccard similarity
- Mean top-one confidence
- High-confidence disagreements
- Jensen-Shannon distance between prediction distributions
- Frequent category shifts from the old model to the new model

## Metrics

| Metric | Meaning |
|---|---|
| Top-one agreement | How often both models select the same first category |
| Cohen's kappa | Agreement adjusted for agreement expected by chance |
| Top-three Jaccard similarity | Overlap between the models' top-three category sets |
| Mean confidence | Average confidence of each model's first prediction |
| Jensen-Shannon distance | Difference between overall prediction distributions; lower is more similar |
| High-confidence disagreement | Both models are confident but choose different categories |

Agreement is not the same as accuracy. Add a human-verified label column to calculate accuracy, precision, recall, and F1 scores.

## Current analysis notes

Review these points before treating `script.py` results as final evaluation metrics:

- `top1_match` currently compares confidence columns instead of top-category columns.
- `run.py` stores confidence on a `0-100` scale, but the disagreement threshold is `0.80`. Use `80.0` for an 80% threshold, or store confidence on a `0-1` scale.
- The distribution comparison uses old-model categories as its index, so categories predicted only by the new model are omitted.

These issues do not affect prediction generation, but they can affect the reported comparison statistics.

## Publishing on GitHub

The trained models are approximately **900 MB** in total, and individual weight files may exceed GitHub's regular file-size limit. Use [Git LFS](https://git-lfs.com/) or publish the weights separately, for example on the Hugging Face Hub.

```bash
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
```

Also check that input and generated CSV files contain no confidential or personally identifiable information before publishing.

## Future improvements

- Add a labeled test set and calculate accuracy, precision, recall, and F1
- Move paths and settings to command-line arguments or a configuration file
- Connect the HTML template to Flask or FastAPI
- Add confusion matrices and per-category evaluation
- Add automated tests and a pinned `requirements.txt`
