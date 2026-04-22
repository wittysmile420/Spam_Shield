# Demo Walkthrough - SMS Spam Shield

Use this flow for a 5-7 minute strong demo.

## 1) Open with Objective and Success Criteria
Say:
- We built an end-to-end SMS spam detection product.
- Success targets are `F1 > 0.95` and `ROC AUC > 0.98`.
- The model is evaluated against Naive Bayes and Linear SVM baselines.

## 2) Project Tour (30-45 sec)
Show:
- `train_model.py`
- `app.py`
- `artifacts/metrics.json`
- `docs/model_report.md`
- `docs/ARCHITECTURE.md`

## 3) Run the Reproducible Pipeline
In terminal:

```powershell
.\run_reproducible_pipeline.bat
```

Explain:
- Runs tests
- Trains model with quality gate
- Generates formal report automatically

## 4) Live Message Demo (Threshold Tradeoff)
In `Live Message Lab`:
1. Enter a spam-like message.
2. Show predicted class + spam probability.
3. Move threshold from `0.40` to `0.60`.
4. Explain business tradeoff:
   - Lower threshold: more spam caught, more false alerts.
   - Higher threshold: fewer false alerts, more spam misses.

## 5) Interpretability Demo
Still in live/dashboard flow:
1. Show matched signal tokens under prediction.
2. Open `Model Dashboard` and show top spam/ham keyword charts.
3. Explain two concrete examples of why these tokens influence decisions.

## 6) Batch Narrative (Upload -> Score -> Export)
In `Batch CSV Studio`:
1. Upload CSV with message column.
2. Run scoring.
3. Show `spam_probability` and `prediction`.
4. Point out threshold delta and flipped-row count.
5. Export `spam_predictions.csv`.

## 7) Evaluation + Error Analysis
Show `docs/model_report.md`:
- Baseline comparison table
- Confusion matrix summary
- 10 false positives with reasons
- 10 false negatives with reasons
- Threshold sweep table

## 8) Close with Risk and Production Readiness
Mention:
- Ethics/risk considerations are documented in `docs/ETHICS_AND_RISK.md`.
- Privacy note for uploaded CSVs is included.
- Model retraining is reproducible and gated for quality.
