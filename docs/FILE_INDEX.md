# File Index

## Root
- `spam.csv`: source training dataset
- `spam (1).ipynb`: original experimentation notebook
- `train_model.py`: production training pipeline with quality gates + baselines
- `generate_model_report.py`: generates formal markdown report from metrics
- `app.py`: Streamlit product UI
- `text_preprocessing.py`: shared normalization utilities
- `requirements.txt`: pinned dependency versions
- `run_product.bat`: install -> train -> report -> launch app
- `run_reproducible_pipeline.bat`: one-command install + tests + train + report

## Tests
- `tests/test_app_utils.py`: CSV parsing, threshold behavior, artifact compatibility, schema checks
- `tests/test_train_model.py`: dataset validation, threshold sweep, quality-gate checks

## Artifacts
- `artifacts/spam_model.joblib`: latest model payload
- `artifacts/metrics.json`: latest evaluation + analysis payload
- `artifacts/spam_model_<timestamp>.joblib`: versioned model history
- `artifacts/metrics_<timestamp>.json`: versioned metric history

## Documentation
- `README.md`: objective, setup, reproducibility, docs map
- `docs/FULL_PROJECT_REPORT.md`: complete master report from fundamentals to full repo
- `docs/model_report.md`: baseline comparison and error analysis report
- `docs/DEMO_WALKTHROUGH.md`: presentation script
- `docs/DEVELOPER_GUIDE.md`: implementation and maintenance details
- `docs/ARCHITECTURE.md`: architecture diagram
- `docs/ETHICS_AND_RISK.md`: risk and privacy notes
- `docs/SUBMISSION_CHECKLIST.md`: final packaging checklist
- `docs/USER_GUIDE.md`: end-user flow
