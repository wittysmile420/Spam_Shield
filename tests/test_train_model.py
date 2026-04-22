from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

import train_model


class TrainModelTests(unittest.TestCase):
    def _make_tmp_dir(self) -> Path:
        root = Path("tests/.tmp")
        root.mkdir(parents=True, exist_ok=True)
        tmp_dir = root / f"train_{uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    def test_load_dataset_accepts_v1_v2_schema(self) -> None:
        csv_data = "v1,v2\nham,hello there\nspam,claim now\n"
        tmp_dir = self._make_tmp_dir()
        try:
            path = tmp_dir / "sample.csv"
            path.write_text(csv_data, encoding="utf-8")
            df = train_model.load_dataset(path)
            self.assertListEqual(list(df.columns), ["label", "message"])
            self.assertEqual(df["label"].tolist(), [0, 1])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_dataset_rejects_invalid_schema(self) -> None:
        csv_data = "text,target\nhi,0\n"
        tmp_dir = self._make_tmp_dir()
        try:
            path = tmp_dir / "bad.csv"
            path.write_text(csv_data, encoding="utf-8")
            with self.assertRaises(ValueError):
                train_model.load_dataset(path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_validate_training_data_rejects_single_class(self) -> None:
        df = pd.DataFrame({"label": [0, 0, 0], "message": ["a", "b", "c"]})
        with self.assertRaises(ValueError):
            train_model.validate_training_data(df)

    def test_compute_threshold_sweep_returns_expected_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        sweep = train_model.compute_threshold_sweep(y_true=y_true, y_prob=y_prob, thresholds=(0.5,))
        self.assertEqual(len(sweep), 1)
        self.assertIn("precision", sweep[0])
        self.assertIn("spam_alert_rate", sweep[0])

    def test_enforce_f1_quality_gate_detects_regression(self) -> None:
        previous_metrics = {"f1_score": 0.95}
        with self.assertRaises(RuntimeError):
            train_model.enforce_f1_quality_gate(current_f1=0.90, previous_metrics=previous_metrics, max_drop=0.01)

    def test_enforce_f1_quality_gate_passes_when_improved(self) -> None:
        previous_metrics = {"f1_score": 0.90}
        gate = train_model.enforce_f1_quality_gate(current_f1=0.95, previous_metrics=previous_metrics, max_drop=0.01)
        self.assertTrue(gate["passed"])


if __name__ == "__main__":
    unittest.main()
