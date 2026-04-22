# Developer Guide - SMS Spam Shield

## Architecture Overview
- `train_model.py`: training entrypoint with validation, baseline comparison, and quality gates
- `generate_model_report.py`: converts `artifacts/metrics.json` into `docs/model_report.md`
- `app.py`: Streamlit UI for live prediction, batch scoring, threshold analysis, and dashboards
- `text_preprocessing.py`: shared normalization logic for train/inference compatibility
- `artifacts/`: latest + versioned model/metrics payloads

## Training Pipeline
1. Load `spam.csv` (`v1/v2` or `label/message`)
2. Validate schema and class balance
3. Normalize text using shared preprocessing
4. Split train/test with fixed seed and stratification
5. Tune Logistic Regression (`C`) via deterministic StratifiedKFold
6. Evaluate:
   - production model metrics
   - Naive Bayes baseline
   - Linear SVM baseline
7. Generate:
   - ROC + confusion matrix
   - threshold sweep
   - top keyword signals
   - false-positive/false-negative examples
8. Apply F1 quality gate vs previous best
9. Export latest + timestamped artifacts

## Reproducibility
- Seed: `42`
- Deterministic split/CV settings are stored in metrics
- Pinned package versions in `requirements.txt`
- Reproduce everything with:

```powershell
.\run_reproducible_pipeline.bat
```

## Quality Gates and Checks
Before training:
- Schema validation (`label/message` contract)
- Class-balance validation (minority class threshold)

After training:
- Fail run if current F1 drops more than configured margin vs previous best

## Test Suite
Location: `tests/`

Coverage focus:
- CSV parsing edge cases
- Threshold decision behavior
- Artifact load compatibility handling
- Prediction output schema checks
- Training quality-gate behavior

Run tests:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Notebook Integration
`spam (1).ipynb` experiments were reused to formalize:
- baseline family (`MultinomialNB`, `LogisticRegression`, `SVM`)
- LR hyperparameter search pattern

These are now encoded in `train_model.py` as reproducible production steps.
