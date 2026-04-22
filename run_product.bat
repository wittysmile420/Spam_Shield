@echo off
setlocal

echo [1/4] Installing dependencies...
call .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency install failed.
  exit /b 1
)

echo [2/4] Training model on spam.csv...
call .\.venv\Scripts\python.exe train_model.py
if errorlevel 1 (
  echo Training failed.
  exit /b 1
)

echo [3/4] Generating latest model report...
call .\.venv\Scripts\python.exe generate_model_report.py
if errorlevel 1 (
  echo Report generation failed.
  exit /b 1
)

echo [4/4] Launching frontend...
echo Open the URL shown below in your browser.
call .\.venv\Scripts\python.exe -m streamlit run app.py

endlocal
