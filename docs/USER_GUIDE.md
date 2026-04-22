# User Guide - SMS Spam Shield

## What This Product Does
SMS Spam Shield predicts whether a message is spam or ham (not spam).

## Start The Product
1. Double-click `run_product.bat`, or in PowerShell run `.\run_product.bat`.
2. Wait for setup, training, and frontend launch.
3. Open the local URL printed by Streamlit.

## Live Predictor
1. Go to `Live Predictor` tab.
2. Enter any SMS text.
3. Click `Analyze Message`.
4. Read label and confidence score.

## Batch CSV Prediction
1. Go to `Batch CSV` tab.
2. Upload CSV file.
3. If CSV has a `message` column, it will be used automatically.
4. Otherwise, first column is treated as text input.
5. Download scored CSV with:
   - `spam_probability`
   - `prediction`

## Model Dashboard
Use this tab to understand model quality:
- ROC curve
- Confusion matrix
- Top spam/ham tokens

## Troubleshooting
- If model is missing, run training again:
  `& .\.venv\Scripts\python.exe train_model.py`
- If app fails to start, reinstall dependencies:
  `& .\.venv\Scripts\python.exe -m pip install -r requirements.txt`
