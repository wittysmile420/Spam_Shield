# Final Submission Checklist

## Core Deliverables
- [ ] `README.md` includes problem statement, objective, and success criteria
- [ ] `train_model.py` runs successfully on `spam.csv`
- [ ] `app.py` launches and all tabs are working
- [ ] `artifacts/spam_model.joblib` and `artifacts/metrics.json` are present

## Evaluation and Reporting
- [ ] `docs/model_report.md` generated and up to date
- [ ] Baseline comparison table includes Logistic, Naive Bayes, Linear SVM
- [ ] Error analysis includes 10 false positives and 10 false negatives with reasons
- [ ] Threshold sweep results included and explained

## Reproducibility and Quality
- [ ] Dependencies are pinned in `requirements.txt`
- [ ] `.\run_reproducible_pipeline.bat` completes successfully
- [ ] Test suite passes (`unittest`)
- [ ] F1 regression quality gate is active in training

## Presentation Readiness
- [ ] `docs/DEMO_WALKTHROUGH.md` followed and rehearsed
- [ ] Threshold tradeoff demo prepared
- [ ] Top keyword interpretability explanation prepared
- [ ] Batch upload to export narrative prepared

## Ethics and Risk
- [ ] `docs/ETHICS_AND_RISK.md` reviewed
- [ ] Bias/language limitations explained
- [ ] Misclassification risks explained
- [ ] CSV privacy note included

## Bonus Recommendations (for extra polish)
- [ ] Add CI workflow to run tests + training dry-run on push
- [ ] Add data drift checks for incoming batch distributions
- [ ] Add model card version history in `docs/model_report.md`
