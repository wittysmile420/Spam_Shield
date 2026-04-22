from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import app


class UploadStub:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def getvalue(self) -> bytes:
        return self._raw


class DummyModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, texts):  # noqa: ANN001
        rows = [[1.0 - self.probability, self.probability] for _ in texts]
        return np.array(rows, dtype=float)


class AppUtilsTests(unittest.TestCase):
    def _make_tmp_dir(self) -> Path:
        root = Path("tests/.tmp")
        root.mkdir(parents=True, exist_ok=True)
        tmp_dir = root / f"app_{uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    def test_read_uploaded_csv_supports_utf8_sig(self) -> None:
        raw = "message,label\nhello,ham\n".encode("utf-8-sig")
        df, encoding = app.read_uploaded_csv(UploadStub(raw))
        self.assertEqual(encoding, "utf-8-sig")
        self.assertListEqual(list(df.columns), ["message", "label"])

    def test_read_uploaded_csv_supports_cp1252(self) -> None:
        raw = "message\ncaf\xe9\n".encode("cp1252")
        df, encoding = app.read_uploaded_csv(UploadStub(raw))
        self.assertEqual(encoding, "cp1252")
        self.assertIn("message", df.columns)

    def test_read_uploaded_csv_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            app.read_uploaded_csv(UploadStub(b""))

    def test_predict_text_threshold_behavior(self) -> None:
        model = DummyModel(probability=0.8)
        label_a, confidence_a, prob_a = app.predict_text(model, "free offer", threshold=0.5)
        label_b, confidence_b, prob_b = app.predict_text(model, "free offer", threshold=0.9)

        self.assertEqual(label_a, "Spam")
        self.assertAlmostEqual(confidence_a, 0.8)
        self.assertAlmostEqual(prob_a, 0.8)
        self.assertEqual(label_b, "Ham")
        self.assertAlmostEqual(confidence_b, 0.2)
        self.assertAlmostEqual(prob_b, 0.8)

    def test_load_model_payload_success(self) -> None:
        tmp_dir = self._make_tmp_dir()
        try:
            model_path = tmp_dir / "model.joblib"
            payload = {"model": "dummy", "training_metrics": {"accuracy": 1.0}}
            joblib.dump(payload, model_path)
            loaded = app.load_model_payload(str(model_path), model_path.stat().st_mtime)
            self.assertIn("training_metrics", loaded)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_model_payload_raises_runtime_error_on_compatibility_issue(self) -> None:
        with patch.object(app.joblib, "load", side_effect=AttributeError("old artifact")):
            with self.assertRaises(RuntimeError):
                app.load_model_payload("fake.joblib", 0.0)

    def test_validate_prediction_schema(self) -> None:
        valid = pd.DataFrame(
            {"spam_probability": [0.1, 0.9], "prediction": ["Ham", "Spam"], "message": ["a", "b"]}
        )
        app.validate_prediction_schema(valid)

        invalid_prob = pd.DataFrame({"spam_probability": [1.2], "prediction": ["Spam"]})
        with self.assertRaises(ValueError):
            app.validate_prediction_schema(invalid_prob)

        invalid_label = pd.DataFrame({"spam_probability": [0.7], "prediction": ["Junk"]})
        with self.assertRaises(ValueError):
            app.validate_prediction_schema(invalid_label)

    def test_load_metrics_fallback_uses_metrics_file(self) -> None:
        tmp_dir = self._make_tmp_dir()
        try:
            fake_metrics = tmp_dir / "metrics.json"
            fake_metrics.write_text(json.dumps({"f1_score": 0.99}), encoding="utf-8")
            with patch.object(app, "METRICS_PATH", fake_metrics):
                loaded = app.load_metrics_fallback({})
                self.assertEqual(loaded["f1_score"], 0.99)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
