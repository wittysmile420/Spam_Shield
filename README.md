# SMS Spam Shield

SMS Spam Shield is an end-to-end ML product that trains on `spam.csv`, exports versioned artifacts, and serves predictions through a Streamlit app for both live and batch workflows.

## Problem Statement
Organizations and users receive large volumes of SMS where spam can include fraud, phishing, and unwanted promotions.  
The project goal is to classify messages as `spam` vs `ham` with high reliability while keeping false alarms low enough for practical usage.

## Project Objective and Success Criteria
Primary objective:
- Build a deployable spam classifier with strong recall/precision balance and explainable outputs.

Target thresholds:
- `F1 > 0.95`
- `ROC AUC > 0.98`

Why this matters in real SMS filtering:
- High false positives block legitimate messages and reduce trust.
- High false negatives allow harmful spam through.
- Threshold tuning allows teams to adapt model behavior to business risk appetite.

## Current Capabilities
- Hybrid feature engineering: word + character TF-IDF
- Tuned Logistic Regression production model (class-balanced)
- Baseline comparison against:
  - Naive Bayes baseline
  - Linear SVM baseline
- Error analysis with real false-positive/false-negative examples
- Threshold sweep to demonstrate precision/recall business tradeoff
- Versioned artifacts:
  - `artifacts/spam_model.joblib`
  - `artifacts/metrics.json`
  - timestamped model + metric history

## One-Command Reproducibility
Run the full reproducible pipeline (install, tests, train, report):

```powershell
.\run_reproducible_pipeline.bat
```

This command provides:
- Fixed random seed and deterministic split/CV settings
- Input schema and class-balance validation before training
- Post-training F1 regression quality gate
- Auto-generation of `docs/model_report.md`

## Quick Start (Product Demo)
### 1) Install dependencies
```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Train + generate report
```powershell
& .\.venv\Scripts\python.exe train_model.py
& .\.venv\Scripts\python.exe generate_model_report.py
```

### 3) Launch frontend
```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

### 4) One-click launch (Windows)
```powershell
.\run_product.bat
```

## Documentation
- `docs/FULL_PROJECT_REPORT.md`: complete zero-to-end codebase report (architecture + all files)
- `docs/model_report.md`: evaluation report, baseline table, error analysis, threshold sweep
- `docs/DEMO_WALKTHROUGH.md`: presentation flow
- `docs/DEVELOPER_GUIDE.md`: pipeline and engineering details
- `docs/ARCHITECTURE.md`: architecture diagram
- `docs/ETHICS_AND_RISK.md`: bias/risk/privacy notes
- `docs/SUBMISSION_CHECKLIST.md`: final packaging checklist
- `docs/FILE_INDEX.md`: complete file map

## Data Contract
Training file must be `spam.csv` with either:
- `v1`, `v2` columns, or
- `label`, `message` columns

Label values must be `ham` / `spam`.

## Notebook Reuse Note
Baseline model family and LR hyperparameter-grid pattern were adapted from `spam (1).ipynb` and formalized into the production training pipeline + report workflow.
