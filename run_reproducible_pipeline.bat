@echo off
setlocal

set PYTHONHASHSEED=42

echo [1/4] Installing pinned dependencies...
call .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency install failed.
  exit /b 1
)

echo [2/4] Running lightweight test suite...
call .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
if errorlevel 1 (
  echo Tests failed.
  exit /b 1
)

echo [3/4] Training model with quality gates...
call .\.venv\Scripts\python.exe train_model.py
if errorlevel 1 (
  echo Training failed.
  exit /b 1
)

echo [4/4] Generating model report...
call .\.venv\Scripts\python.exe generate_model_report.py
if errorlevel 1 (
  echo Report generation failed.
  exit /b 1
)

echo Reproducible pipeline complete.
endlocal
