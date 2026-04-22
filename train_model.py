from __future__ import annotations

"""Train and export a production-ready SMS spam detection model.

Outputs:
- artifacts/spam_model.joblib (latest model payload)
- artifacts/metrics.json (latest metrics payload)
- timestamped versioned copies for traceability
"""

import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from text_preprocessing import normalize_text

DATA_PATH = Path("spam.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "spam_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CLASS_BALANCE_MIN_RATIO = 0.02
MAX_F1_REGRESSION = 0.01
ERROR_EXAMPLES_PER_CLASS = 10
SUCCESS_THRESHOLDS = {"f1_score": 0.95, "roc_auc": 0.98}


def set_reproducible_seeds(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)


def get_package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn_version,
        "joblib": joblib.__version__,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
    elif {"label", "message"}.issubset(df.columns):
        df = df[["label", "message"]].copy()
    else:
        raise ValueError("CSV must contain either columns ['v1','v2'] or ['label','message']")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    allowed_labels = {"ham", "spam"}
    unknown_labels = sorted(set(df["label"]) - allowed_labels)
    if unknown_labels:
        raise ValueError(
            f"Unsupported labels found: {unknown_labels}. Allowed labels are only 'ham' and 'spam'."
        )

    df = df[df["label"].isin(allowed_labels)].copy()
    df["message"] = df["message"].fillna("").astype(str).str.strip()
    df = df[df["message"] != ""].copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return df.reset_index(drop=True)


def validate_training_data(df: pd.DataFrame, min_class_ratio: float = CLASS_BALANCE_MIN_RATIO) -> dict[str, Any]:
    required_cols = {"label", "message"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")

    if df["label"].nunique() < 2:
        raise ValueError("Training requires both classes ('ham' and 'spam').")

    ham_count = int((df["label"] == 0).sum())
    spam_count = int((df["label"] == 1).sum())
    total = int(len(df))
    minority_ratio = min(ham_count, spam_count) / float(total)
    if minority_ratio < min_class_ratio:
        raise ValueError(
            f"Class imbalance too high: minority ratio {minority_ratio:.4f} < {min_class_ratio:.4f}"
        )

    return {
        "ham_count": ham_count,
        "spam_count": spam_count,
        "minority_ratio": round(float(minority_ratio), 4),
        "total_rows": total,
    }


def build_feature_union() -> FeatureUnion:
    word_features = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=20000,
        sublinear_tf=True,
    )
    char_features = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    return FeatureUnion(transformer_list=[("word", word_features), ("char", char_features)])


def build_logistic_pipeline() -> Pipeline:
    """Primary production pipeline used by the app."""
    return Pipeline(
        steps=[
            ("features", build_feature_union()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_notebook_baseline_models() -> dict[str, Pipeline]:
    """Baseline family adapted from experiments in spam (1).ipynb."""

    def make_word_vectorizer(max_features: int = 3000) -> TfidfVectorizer:
        return TfidfVectorizer(
            preprocessor=normalize_text,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=max_features,
            sublinear_tf=True,
        )

    return {
        "Naive Bayes baseline": Pipeline(
            steps=[("features", make_word_vectorizer()), ("clf", MultinomialNB())]
        ),
        "Linear SVM baseline": Pipeline(
            steps=[
                ("features", make_word_vectorizer()),
                ("clf", LinearSVC(random_state=RANDOM_SEED)),
            ]
        ),
    }


def score_model_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
    }


def evaluate_model(
    name: str,
    model: Pipeline,
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)
    else:
        y_score = y_pred.astype(float)

    scored = score_model_predictions(
        y_true=y_test.to_numpy(),
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
    )
    return {"model": name, **scored}


def extract_top_keywords(model: Pipeline, top_n: int = 15) -> dict[str, list[dict[str, float]]]:
    features_union: FeatureUnion = model.named_steps["features"]
    clf: LogisticRegression = model.named_steps["clf"]

    word_vectorizer: TfidfVectorizer = dict(features_union.transformer_list)["word"]
    word_names = word_vectorizer.get_feature_names_out()
    word_count = len(word_names)
    coefs = clf.coef_[0][:word_count]

    top_spam_idx = coefs.argsort()[-top_n:][::-1]
    top_ham_idx = coefs.argsort()[:top_n]
    return {
        "spam": [
            {"token": str(word_names[i]), "weight": round(float(coefs[i]), 4)} for i in top_spam_idx
        ],
        "ham": [
            {"token": str(word_names[i]), "weight": round(float(coefs[i]), 4)} for i in top_ham_idx
        ],
    }


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "spam_alert_rate": round(float(np.mean(y_pred)), 4),
            }
        )
    return rows


def infer_error_reason(
    message: str,
    error_type: str,
    spam_probability: float,
    top_keywords: dict[str, list[dict[str, float]]],
) -> str:
    normalized = normalize_text(message)
    spam_tokens = [item["token"] for item in top_keywords.get("spam", [])]
    ham_tokens = [item["token"] for item in top_keywords.get("ham", [])]

    matched_spam = [token for token in spam_tokens if token and token in normalized][:3]
    matched_ham = [token for token in ham_tokens if token and token in normalized][:3]
    near_boundary = abs(spam_probability - 0.5) <= 0.1

    if error_type == "false_positive":
        if matched_spam:
            return f"Ham message contains strong spam-like terms: {', '.join(matched_spam)}."
        if near_boundary:
            return "Borderline score near threshold made the decision unstable."
        return "Message style looked promotional or numeric despite being legitimate."

    if matched_ham:
        return f"Spam message looks conversational with ham-like terms: {', '.join(matched_ham)}."
    if near_boundary:
        return "Borderline score near threshold caused a miss."
    return "Spam cues were weak compared with conversational language patterns."


def collect_error_examples(
    test_messages: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    top_keywords: dict[str, list[dict[str, float]]],
    limit: int = ERROR_EXAMPLES_PER_CLASS,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {"false_positives": [], "false_negatives": []}

    rows = pd.DataFrame(
        {
            "message": test_messages.reset_index(drop=True),
            "actual": y_true,
            "predicted": y_pred,
            "spam_probability": y_prob,
        }
    )
    false_pos = rows[(rows["actual"] == 0) & (rows["predicted"] == 1)].sort_values(
        "spam_probability", ascending=False
    )
    false_neg = rows[(rows["actual"] == 1) & (rows["predicted"] == 0)].sort_values(
        "spam_probability", ascending=True
    )

    for _, row in false_pos.head(limit).iterrows():
        message = str(row["message"])
        prob = float(row["spam_probability"])
        examples["false_positives"].append(
            {
                "message": message,
                "spam_probability": round(prob, 4),
                "reason": infer_error_reason(message, "false_positive", prob, top_keywords),
            }
        )

    for _, row in false_neg.head(limit).iterrows():
        message = str(row["message"])
        prob = float(row["spam_probability"])
        examples["false_negatives"].append(
            {
                "message": message,
                "spam_probability": round(prob, 4),
                "reason": infer_error_reason(message, "false_negative", prob, top_keywords),
            }
        )

    return examples


def load_previous_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def enforce_f1_quality_gate(
    current_f1: float, previous_metrics: dict[str, Any], max_drop: float = MAX_F1_REGRESSION
) -> dict[str, Any]:
    previous_f1_raw = previous_metrics.get("f1_score")
    previous_f1 = float(previous_f1_raw) if previous_f1_raw is not None else None
    gate = {
        "previous_best_f1": round(previous_f1, 4) if previous_f1 is not None else None,
        "current_f1": round(float(current_f1), 4),
        "max_allowed_drop": round(float(max_drop), 4),
        "passed": True,
    }

    if previous_f1 is not None and current_f1 < (previous_f1 - max_drop):
        gate["passed"] = False
        raise RuntimeError(
            f"Quality gate failed: current F1={current_f1:.4f} is below previous F1={previous_f1:.4f} "
            f"by more than allowed drop={max_drop:.4f}."
        )

    return gate


def evaluate_success_criteria(metrics: dict[str, float]) -> dict[str, Any]:
    status: dict[str, Any] = {"thresholds": SUCCESS_THRESHOLDS.copy(), "results": {}, "passed_all": True}
    for metric_name, threshold in SUCCESS_THRESHOLDS.items():
        actual = float(metrics[metric_name])
        passed = actual >= threshold
        status["results"][metric_name] = {
            "actual": round(actual, 4),
            "target": threshold,
            "passed": passed,
        }
        if not passed:
            status["passed_all"] = False
    return status


def train() -> None:
    set_reproducible_seeds(RANDOM_SEED)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    previous_metrics = load_previous_metrics(METRICS_PATH)

    df = load_dataset(DATA_PATH)
    validation_summary = validate_training_data(df)
    df["cleaned"] = df["message"].apply(normalize_text)

    features = df[["message", "cleaned"]]
    labels = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    x_train_clean = x_train["cleaned"]
    x_test_clean = x_test["cleaned"]

    base_pipeline = build_logistic_pipeline()
    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid={"clf__C": [4, 8, 12]},
        scoring="f1",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=1,
        refit=True,
    )
    grid.fit(x_train_clean, y_train)
    model: Pipeline = grid.best_estimator_

    y_pred = model.predict(x_test_clean)
    y_prob = model.predict_proba(x_test_clean)[:, 1]
    y_true = y_test.to_numpy()

    logistic_metrics = score_model_predictions(y_true=y_true, y_pred=np.asarray(y_pred), y_score=np.asarray(y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    top_keywords = extract_top_keywords(model)

    baseline_results = [dict(model="Logistic (current)", **logistic_metrics)]
    for baseline_name, baseline_model in build_notebook_baseline_models().items():
        baseline_results.append(
            evaluate_model(
                name=baseline_name,
                model=baseline_model,
                x_train=x_train_clean,
                y_train=y_train,
                x_test=x_test_clean,
                y_test=y_test,
            )
        )

    threshold_sweep = compute_threshold_sweep(y_true=y_true, y_prob=np.asarray(y_prob))
    oof_model = build_logistic_pipeline()
    oof_model.set_params(**grid.best_params_)
    oof_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = cross_val_predict(
        oof_model,
        df["cleaned"],
        df["label"],
        cv=oof_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    oof_pred = (oof_prob >= 0.5).astype(int)
    error_analysis = collect_error_examples(
        test_messages=df["message"],
        y_true=df["label"].to_numpy(),
        y_pred=oof_pred,
        y_prob=oof_prob,
        top_keywords=top_keywords,
        limit=ERROR_EXAMPLES_PER_CLASS,
    )

    quality_gate = enforce_f1_quality_gate(current_f1=float(logistic_metrics["f1_score"]), previous_metrics=previous_metrics)
    success_criteria = evaluate_success_criteria(logistic_metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics: dict[str, Any] = {
        "model_version": f"spam_shield_{timestamp}",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_params": grid.best_params_,
        **logistic_metrics,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(float(v), 6) for v in fpr.tolist()],
            "tpr": [round(float(v), 6) for v in tpr.tolist()],
        },
        "top_keywords": top_keywords,
        "threshold_sweep": threshold_sweep,
        "baseline_comparison": baseline_results,
        "error_analysis_source": "out_of_fold_predictions_at_threshold_0.50",
        "error_analysis": error_analysis,
        "quality_gate": quality_gate,
        "success_criteria": success_criteria,
        "data_validation": validation_summary,
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "package_versions": get_package_versions(),
        },
        "notebook_reuse_notes": "Baseline model family and LR C-grid adapted from spam (1).ipynb.",
    }

    versioned_model_path = ARTIFACT_DIR / f"spam_model_{timestamp}.joblib"
    versioned_metrics_path = ARTIFACT_DIR / f"metrics_{timestamp}.json"
    payload = {
        "model": model,
        "label_map": {0: "Ham", 1: "Spam"},
        "training_metrics": metrics,
    }

    joblib.dump(payload, versioned_model_path)
    versioned_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    latest_model_saved = True
    latest_metrics_saved = True
    try:
        joblib.dump(payload, MODEL_PATH)
    except PermissionError:
        latest_model_saved = False
        print(
            f"Warning: Could not overwrite latest model at {MODEL_PATH}. "
            "A process may be locking the file. Versioned model was still saved."
        )

    try:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except PermissionError:
        latest_metrics_saved = False
        print(
            f"Warning: Could not overwrite latest metrics at {METRICS_PATH}. "
            "Versioned metrics were still saved."
        )

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    if latest_model_saved:
        print(f"Saved model: {MODEL_PATH}")
    else:
        print(f"Latest model not updated (locked). Use versioned model: {versioned_model_path}")
    print(f"Saved versioned model: {versioned_model_path}")
    if latest_metrics_saved:
        print(f"Saved metrics: {METRICS_PATH}")
    else:
        print(f"Latest metrics not updated (locked). Use versioned metrics: {versioned_metrics_path}")
    print(f"Saved versioned metrics: {versioned_metrics_path}")


if __name__ == "__main__":
    train()
