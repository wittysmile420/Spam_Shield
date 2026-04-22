from __future__ import annotations

"""Generate a deep technical project report from artifacts/metrics.json."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(".")
METRICS_PATH = Path("artifacts/metrics.json")
REPORT_PATH = Path("docs/model_report.md")
FULL_REPORT_PATH = Path("docs/FULL_PROJECT_REPORT.md")

TEAM_CONTRIBUTIONS = {
    "Avi": [
        "Owned dataset-level exploration and early model-family experimentation in notebook workflow.",
        "Benchmarked classical text classifiers and validated baseline behavior before productionization.",
        "Helped define feature and metric expectations used in the final production pipeline.",
    ],
    "Rhythm": [
        "Owned production training pipeline hardening in train_model.py.",
        "Implemented deterministic training setup, schema checks, class-balance checks, and quality gates.",
        "Built baseline-comparison, threshold-sweep, and error-analysis integration into metrics payload.",
    ],
    "Harshit": [
        "Owned product-side inference experience in app.py.",
        "Implemented live prediction workflow, batch CSV scoring/export, dashboard visualization, and schema validation.",
        "Integrated model explainability surfaces (token signals and threshold tradeoff) into UI narrative.",
    ],
    "Sarthak": [
        "Owned end-to-end integration, release readiness, and documentation packaging.",
        "Built reproducibility automation scripts and report-generation workflow.",
        "Consolidated architecture, ethics/risk, developer, demo, and submission documentation.",
    ],
}

KEY_FILE_PURPOSES = {
    "README.md": "Project summary, objective, setup, and reproducibility entry points.",
    "requirements.txt": "Pinned dependency versions for repeatable runs.",
    "train_model.py": "Core ML training/evaluation pipeline with baselines, quality gates, and artifact writing.",
    "text_preprocessing.py": "Shared normalization logic to keep train/inference behavior aligned.",
    "app.py": "Streamlit product UI for live prediction, batch scoring, analytics, and exports.",
    "generate_model_report.py": "Automated report generator for deep technical documentation.",
    "run_product.bat": "Single-click product run sequence (install -> train -> report -> app).",
    "run_reproducible_pipeline.bat": "Deterministic CI-like pipeline (install -> tests -> train -> report).",
    "spam.csv": "Primary labeled dataset.",
    "spam (1).ipynb": "Early experimentation notebook used as baseline inspiration.",
    "tests/test_app_utils.py": "App utility tests: CSV parsing, threshold behavior, artifact/schema safety.",
    "tests/test_train_model.py": "Training tests: dataset validation and F1 quality gate behavior.",
    "docs/ARCHITECTURE.md": "Mermaid architecture and design notes.",
    "docs/DEVELOPER_GUIDE.md": "Developer-oriented system and pipeline guidance.",
    "docs/DEMO_WALKTHROUGH.md": "Presentation/demo narrative.",
    "docs/ETHICS_AND_RISK.md": "Bias, safety, and privacy considerations.",
    "docs/FILE_INDEX.md": "Project map reference.",
    "docs/SUBMISSION_CHECKLIST.md": "Submission readiness checklist.",
    "docs/USER_GUIDE.md": "End-user operation guide.",
    "docs/model_report.md": "Auto-generated technical report (this output).",
    "docs/FULL_PROJECT_REPORT.md": "Mirror of the technical report for master-document access.",
    "artifacts/spam_model.joblib": "Latest model payload for app inference.",
    "artifacts/metrics.json": "Latest metrics payload used by app/reporting.",
}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header, sep, body]) if rows else "\n".join([header, sep, "| " + " | ".join([""] * len(headers)) + " |"])


def resolve_metrics_path() -> Path:
    candidates = list(Path("artifacts").glob("metrics*.json"))
    if not candidates:
        raise FileNotFoundError("No metrics JSON files found in artifacts/ directory.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_metrics() -> tuple[Path, dict[str, Any]]:
    metrics_path = METRICS_PATH if METRICS_PATH.exists() else resolve_metrics_path()
    latest = resolve_metrics_path()
    if latest.stat().st_mtime > metrics_path.stat().st_mtime:
        metrics_path = latest
    return metrics_path, json.loads(metrics_path.read_text(encoding="utf-8"))


def list_repo_files(root: Path = ROOT_DIR) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".venv/"):
            continue
        files.append(rel)
    return sorted(files)


def extract_function_index(file_path: Path) -> list[tuple[str, int]]:
    if not file_path.exists():
        return []
    functions: list[tuple[str, int]] = []
    pattern = re.compile(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\(")
    for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = pattern.match(line.strip())
        if match:
            functions.append((match.group(1), line_no))
    return functions


def extract_test_cases(file_path: Path) -> list[str]:
    if not file_path.exists():
        return []
    cases: list[str] = []
    pattern = re.compile(r"^\s*def\s+(test_[a-zA-Z0-9_]+)\(")
    for line in file_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            cases.append(match.group(1))
    return cases


def render_requirements_table(path: Path = Path("requirements.txt")) -> str:
    if not path.exists():
        return "_requirements.txt not found._"
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if "==" in cleaned:
            name, version = cleaned.split("==", maxsplit=1)
            rows.append([name.strip(), version.strip()])
        else:
            rows.append([cleaned, "unpinned"])
    return md_table(["Package", "Version"], rows)


def render_baseline_table(metrics: dict[str, Any]) -> str:
    baselines = metrics.get("baseline_comparison", [])
    if not baselines:
        return "_Baseline comparison unavailable._"
    rows: list[list[str]] = []
    for row in baselines:
        rows.append(
            [
                str(row.get("model", "n/a")),
                f"{float(row.get('accuracy', 0.0)):.4f}",
                f"{float(row.get('precision', 0.0)):.4f}",
                f"{float(row.get('recall', 0.0)):.4f}",
                f"{float(row.get('f1_score', 0.0)):.4f}",
                f"{float(row.get('roc_auc', 0.0)):.4f}",
            ]
        )
    return md_table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"], rows)


def render_threshold_sweep_table(metrics: dict[str, Any]) -> str:
    sweep = metrics.get("threshold_sweep", [])
    if not sweep:
        return "_Threshold sweep unavailable._"
    rows: list[list[str]] = []
    for row in sweep:
        rows.append(
            [
                f"{float(row.get('threshold', 0.0)):.2f}",
                f"{float(row.get('precision', 0.0)):.4f}",
                f"{float(row.get('recall', 0.0)):.4f}",
                f"{float(row.get('f1_score', 0.0)):.4f}",
                f"{float(row.get('spam_alert_rate', 0.0)):.4f}",
            ]
        )
    return md_table(["Threshold", "Precision", "Recall", "F1", "Spam Alert Rate"], rows)


def render_top_keyword_table(metrics: dict[str, Any], cls: str, top_n: int = 12) -> str:
    top = metrics.get("top_keywords", {}).get(cls, [])
    if not top:
        return "_Unavailable._"
    rows: list[list[str]] = []
    for item in top[:top_n]:
        rows.append([str(item.get("token", "")), f"{float(item.get('weight', 0.0)):.4f}"])
    return md_table(["Token", "Weight"], rows)


def render_error_examples(title: str, examples: list[dict[str, Any]]) -> str:
    lines = [f"### {title}"]
    if not examples:
        lines.append("No examples available.")
        return "\n".join(lines)
    for idx, item in enumerate(examples[:10], start=1):
        msg = str(item.get("message", "")).replace("\n", " ").strip()
        prob = float(item.get("spam_probability", 0.0))
        reason = str(item.get("reason", "")).strip()
        lines.append(f"{idx}. `{msg}`")
        lines.append(f"   - spam_probability: `{prob:.4f}`")
        lines.append(f"   - diagnostic_reason: {reason}")
    return "\n".join(lines)


def render_repo_inventory(files: list[str]) -> str:
    if not files:
        return "_No files found._"
    return "```text\n" + "\n".join(files) + "\n```"


def render_file_purpose_table(files: list[str]) -> str:
    rows: list[list[str]] = []
    for rel in files:
        if rel in KEY_FILE_PURPOSES:
            rows.append([rel, KEY_FILE_PURPOSES[rel]])
        elif rel.startswith("artifacts/spam_model_") and rel.endswith(".joblib"):
            rows.append([rel, "Timestamped model artifact snapshot for traceability/history."])
        elif rel.startswith("artifacts/metrics_") and rel.endswith(".json"):
            rows.append([rel, "Timestamped metrics artifact snapshot for traceability/history."])
        elif rel.endswith(".pyc"):
            rows.append([rel, "Python bytecode cache file."])
        else:
            rows.append([rel, "Repository file (see module-level docs or folder guide)."])
    return md_table(["File", "Purpose"], rows)


def render_function_index_table(functions: list[tuple[str, int]], file_label: str) -> str:
    if not functions:
        return f"_No functions discovered for {file_label}._"
    rows = [[name, str(line)] for name, line in functions]
    return md_table([f"{file_label} function", "Line"], rows)


def render_test_index_table(test_map: dict[str, list[str]]) -> str:
    rows: list[list[str]] = []
    for rel, tests in test_map.items():
        if not tests:
            rows.append([rel, "No test functions discovered"])
            continue
        rows.append([rel, ", ".join(tests)])
    return md_table(["Test File", "Test Cases"], rows)


def render_team_contributions() -> str:
    lines: list[str] = ["## 18) Team Contributions (4-member collaborative delivery)"]
    for member, points in TEAM_CONTRIBUTIONS.items():
        lines.append(f"### {member}")
        for idx, point in enumerate(points, start=1):
            lines.append(f"{idx}. {point}")
    lines.append(
        "### Combined Outcome"
    )
    lines.append(
        "1. Work was executed as an integrated team pipeline: research -> production training -> product UI -> documentation and release packaging."
    )
    lines.append(
        "2. Final delivery reflects joint ownership of both model quality and deployable product behavior."
    )
    lines.append(
        "3. If your faculty requires a stricter per-person task matrix, update this section with exact commit/task logs from your local collaboration record."
    )
    return "\n".join(lines)


def generate_report(metrics: dict[str, Any], metrics_path: Path, repo_files: list[str]) -> str:
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    tn = int(cm[0][0]) if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = int(cm[0][1]) if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = int(cm[1][0]) if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = int(cm[1][1]) if len(cm) > 1 and len(cm[1]) > 1 else 0

    f1 = float(metrics.get("f1_score", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))
    success = bool(metrics.get("success_criteria", {}).get("passed_all", False))
    quality_gate = bool(metrics.get("quality_gate", {}).get("passed", False))
    reproducibility = metrics.get("reproducibility", {})
    baselines = metrics.get("baseline_comparison", [])
    false_pos = metrics.get("error_analysis", {}).get("false_positives", [])
    false_neg = metrics.get("error_analysis", {}).get("false_negatives", [])

    best_model = "n/a"
    best_f1 = 0.0
    if baselines:
        best_row = max(baselines, key=lambda x: float(x.get("f1_score", 0.0)))
        best_model = str(best_row.get("model", "n/a"))
        best_f1 = float(best_row.get("f1_score", 0.0))

    train_functions = extract_function_index(Path("train_model.py"))
    app_functions = extract_function_index(Path("app.py"))
    report_functions = extract_function_index(Path("generate_model_report.py"))

    test_files = [f for f in repo_files if f.startswith("tests/") and f.endswith(".py")]
    test_map = {test_file: extract_test_cases(Path(test_file)) for test_file in test_files}

    report = f"""# Comprehensive Technical Report - SMS Spam Shield

Generated on: {datetime.now().isoformat(timespec="seconds")}
Loaded metrics source: `{metrics_path.as_posix()}`
Model version: `{metrics.get("model_version", "n/a")}`

## 1) Abstract
SMS Spam Shield is an end-to-end text-classification system that operationalizes SMS spam detection from data ingestion through deployable inference UI.
The system combines machine-learning rigor, reproducibility controls, model risk analysis, and product-grade interaction modes (live scoring + batch processing).
This document is intentionally research-heavy and engineering-deep, covering architecture, algorithms, full repository structure, function-level index, testing strategy, and collaborative team ownership.

## 2) Problem Definition and Research Context
Task type: binary supervised text classification.
Given message text `x`, estimate `P(y=spam|x)`.
Decision rule in production:
`y_hat = 1 if P(y=spam|x) >= tau else 0`,
where `tau` is configurable threshold.

Project success criteria:
1. F1 target: `> 0.95`
2. ROC AUC target: `> 0.98`

Latest achieved:
1. F1: `{f1:.4f}`
2. ROC AUC: `{roc_auc:.4f}`
3. Threshold criteria met: `{success}`
4. Quality gate pass: `{quality_gate}`

## 3) Data Foundation and Contract
Dataset source file: `spam.csv`
Accepted input schema:
1. (`v1`, `v2`) or
2. (`label`, `message`)

Label contract:
1. `ham` -> 0
2. `spam` -> 1
3. non-contract labels rejected

Latest dataset profile:
1. total rows: `{metrics.get("dataset_rows", "n/a")}`
2. train rows: `{metrics.get("train_rows", "n/a")}`
3. test rows: `{metrics.get("test_rows", "n/a")}`
4. validation summary: `{metrics.get("data_validation", {})}`

## 4) Text Preprocessing and Feature Representation
Normalization objective:
preserve spam-discriminative cues while reducing lexical noise.

Key transformations:
1. URL normalization to token placeholder.
2. numeric normalization (`longnumber`, `number`).
3. character filtering with currency preservation.
4. whitespace canonicalization.

Feature stack:
1. Word TF-IDF (1-2 grams).
2. Character TF-IDF (`char_wb`, 3-5 grams).
3. FeatureUnion to merge lexical and subword signals.

Theoretical motivation:
1. word n-grams capture semantic spam keywords.
2. char n-grams capture obfuscation and stylistic spam patterns.
3. hybridization improves robustness under noisy SMS syntax.

## 5) Modeling Strategy and Algorithm Selection
Primary production classifier:
1. Logistic Regression (balanced class weighting, deterministic random state).
2. Hyperparameter tuning over regularization strength `C`.

Baseline suite (formalized from notebook experimentation):
1. Naive Bayes baseline.
2. Linear SVM baseline.
3. Logistic (current production).

Best baseline F1 observed: `{best_model}` with `{best_f1:.4f}`.
Production still uses Logistic because:
1. calibrated probability output supports threshold control in UI workflows.
2. coefficient-level interpretability enables transparent keyword signal explanation.
3. strong ROC-AUC and stable operational behavior across report runs.

{render_baseline_table(metrics)}

## 6) Evaluation Analytics
Confusion matrix (latest holdout):
1. TN: `{tn}`
2. FP: `{fp}`
3. FN: `{fn}`
4. TP: `{tp}`

Operational interpretation:
1. FP contributes to false alarms and user-friction costs.
2. FN contributes to spam leakage and risk exposure.
3. Threshold governance is therefore a policy decision, not only a model score decision.

### Threshold Sweep
{render_threshold_sweep_table(metrics)}

### Top Learned Signals - Spam Class
{render_top_keyword_table(metrics, "spam")}

### Top Learned Signals - Ham Class
{render_top_keyword_table(metrics, "ham")}

### Error Analysis - False Positives
{render_error_examples("False Positives (10 examples)", false_pos)}

### Error Analysis - False Negatives
{render_error_examples("False Negatives (10 examples)", false_neg)}

## 7) Architecture and Runtime Dataflow
```mermaid
flowchart LR
    A[spam.csv] --> B[train_model.py]
    B --> C[Schema and Class-Balance Validation]
    C --> D[Preprocessing + Hybrid TF-IDF Features]
    D --> E[Model Training + Baseline Comparison]
    E --> F[Quality Gate and Metrics Assembly]
    F --> G[artifacts/spam_model.joblib]
    F --> H[artifacts/metrics.json]
    H --> I[generate_model_report.py]
    I --> J[docs/model_report.md]
    G --> K[app.py]
    H --> K
    K --> L[Live Message Lab]
    K --> M[Batch CSV Studio]
    K --> N[Model Dashboard]
    K --> O[Session Log and Export]
```

Design highlights:
1. shared preprocessing module avoids train-serving skew.
2. report generation is data-driven from artifacts payload.
3. product dashboard binds directly to persisted metrics for traceable analytics.

## 8) Function-Level Technical Index
### train_model.py
{render_function_index_table(train_functions, "train_model.py")}

### app.py
{render_function_index_table(app_functions, "app.py")}

### generate_model_report.py
{render_function_index_table(report_functions, "generate_model_report.py")}

## 9) Product Functionality (How It Works for Users)
Live mode:
1. user enters SMS text.
2. model computes spam probability.
3. threshold decides label.
4. confidence and matched signal tokens displayed.

Batch mode:
1. CSV upload with encoding fallbacks.
2. user selects text column.
3. model scores full dataset.
4. output schema validated.
5. downloadable predictions generated.

Dashboard mode:
1. ROC chart.
2. confusion matrix.
3. top signal tokens.
4. threshold tradeoff view.

Session and governance:
1. session log captures inference events.
2. export support enables audit and review workflows.

## 10) Reproducibility and MLOps Controls
Reproducibility payload:
1. random_seed: `{reproducibility.get("random_seed", "n/a")}`
2. test_size: `{reproducibility.get("test_size", "n/a")}`
3. cv_folds: `{reproducibility.get("cv_folds", "n/a")}`
4. package_versions: `{reproducibility.get("package_versions", {})}`

Automation scripts:
1. `run_product.bat` for user-facing launch.
2. `run_reproducible_pipeline.bat` for deterministic validation pipeline.

Quality gates:
1. pre-train schema and class-balance checks.
2. post-train F1 regression gate against prior best.

Dependency pinning:
{render_requirements_table()}

## 11) Testing Engineering
Test inventory summary:
1. test files discovered: `{len(test_files)}`
2. test functions discovered: `{sum(len(v) for v in test_map.values())}`

{render_test_index_table(test_map)}

Test intent coverage:
1. CSV parsing and encoding edge cases.
2. threshold decision correctness.
3. artifact compatibility handling.
4. prediction schema validation.
5. dataset contract validation.
6. quality gate regression enforcement.

## 12) Full Repository Inventory
Total repository files discovered (excluding `.venv`): `{len(repo_files)}`

{render_repo_inventory(repo_files)}

## 13) File-by-File Purpose Map
{render_file_purpose_table(repo_files)}

## 14) Ethics, Risk, and Responsible Use
Risk dimensions:
1. language/domain shift can degrade performance.
2. false positives and false negatives have asymmetric operational impacts.
3. privacy risk exists if uploaded CSVs contain sensitive identifiers.

Mitigation posture:
1. threshold tuning by business policy.
2. periodic retraining and error review.
3. maintain human override for high-stakes decisions.
4. document retention policy for exported prediction files.

## 15) Team Delivery Narrative
This project was built as a coordinated 4-member team with overlapping review loops, not isolated silos.
Each member contributed to both direct implementation and integration validation.
"""

    report += "\n\n" + render_team_contributions() + "\n"
    report += """
## 19) Closing Technical Statement
SMS Spam Shield demonstrates a full-stack ML product pattern:
1. from raw labeled data to validated training pipeline.
2. from benchmark comparison to decision-threshold policy tooling.
3. from model artifact to user-facing inference product.
4. from ad-hoc experimentation to reproducible, test-backed, report-driven delivery.

This report intentionally captures both research depth and engineering completeness so that the project can be defended in technical review, academic viva, and practical deployment discussions.
"""
    return report


def main() -> None:
    metrics_path, metrics = load_metrics()
    repo_files = list_repo_files()
    report_text = generate_report(metrics=metrics, metrics_path=metrics_path, repo_files=repo_files)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    FULL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FULL_REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Loaded metrics: {metrics_path}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved mirror report: {FULL_REPORT_PATH}")


if __name__ == "__main__":
    main()
