from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from pyfedappwrap.learning.base_app import BaseApp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from config import LogisticRegressionAppConfig, LogisticRegressionAppInputConfig, \
    LogisticRegressionAppOutputConfig
from helper import _top_features_html, _coef_table, _write_predictions, _ensure_numeric_matrix, \
    _split_id_and_df, _pop_target


class LogisticRegressionAPP(
    BaseApp[
        LogisticRegressionAppConfig,
        LogisticRegressionAppInputConfig,
        LogisticRegressionAppOutputConfig,
    ]
):
    @override
    def get_test_data(self) -> Optional[dict[str, Path]]:
        return {"data": Path("data/iris.csv")}

    _artifact_path: Optional[Path] = Path("model/model.joblib")
    _artifact: Optional[Dict[str, Any]] = None

    def run_train(self,
                  data: LogisticRegressionAppInputConfig) -> LogisticRegressionAppOutputConfig:
        df: pd.DataFrame = data.data
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Training input 'train' must be a pandas DataFrame.")

        ids, df2 = _split_id_and_df(df)
        X_df, y = _pop_target(df2, self.config.target_column)

        X = _ensure_numeric_matrix(X_df)

        scaler = None
        if bool(self.config.standardize):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        cw = None if str(self.config.class_weight).lower() in {"none", "null", ""} else str(
            self.config.class_weight)

        model = LogisticRegression(
            C=float(self.config.C),
            max_iter=int(self.config.max_iter),
            solver=str(self.config.solver),
            class_weight=cw,
            random_state=int(self.config.random_state),
        )

        self.logger.info(
            f"Training LogisticRegression on {X.shape[0]} samples, {X.shape[1]} features")
        model.fit(X, y)

        self._artifact = {
            "model": model,
            "scaler": scaler,
            "feature_columns": list(X_df.columns),
            "target_column": self.config.target_column,
        }

        coef_df = self._get_model_coef(model, X_df)

        # Basic train metric
        train_acc = float(model.score(X, y))
        self.send_metric("train_accuracy", train_acc, 0)
        self.send_metric("n_samples", float(X.shape[0]), 0)
        self.send_metric("n_features", float(X.shape[1]), 0)

        report = (
            f"Trained LogisticRegression: samples={X.shape[0]}, features={X.shape[1]}, "
            f"train_accuracy={train_acc:.4f}, target_column='{self.config.target_column}'."
        )

        return LogisticRegressionAppOutputConfig(
            coefficients=coef_df,
            explanation=self._get_explanation(coef_df, explain_top_k=self.config.explain_top_k),
            report=report,
        )

    def run_prediction(self,
                       data: LogisticRegressionAppInputConfig) -> LogisticRegressionAppOutputConfig:
        df: pd.DataFrame = data.data
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Prediction input 'features' must be a pandas DataFrame.")

        self._load(str(self._artifact_path))
        artifact = self._artifact
        model: LogisticRegression = artifact["model"]
        scaler = artifact.get("scaler", None)
        feature_cols: List[str] = artifact["feature_columns"]

        ids, df2 = _split_id_and_df(df)

        missing = [c for c in feature_cols if c not in df2.columns]
        if missing:
            raise ValueError(f"Missing feature columns in prediction input: {missing}")

        X_df = df2[feature_cols]
        X = _ensure_numeric_matrix(X_df)

        if bool(self.config.standardize) and scaler is not None:
            X = scaler.transform(X)

        proba = model.predict_proba(X)
        labels = model.predict(X)
        classes = model.classes_

        coef_df = self._get_model_coef(model, X_df)
        report = f"Predicted {len(labels)} samples with LogisticRegression. Probabilities included."
        return LogisticRegressionAppOutputConfig(
            predictions=_write_predictions(ids, labels, proba, classes),
            coefficients=coef_df,
            explanation=self._get_explanation(coef_df, explain_top_k=self.config.explain_top_k),
            report=report,
        )

    def _save(self) -> str:
        self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._artifact, self._artifact_path)
        return str(self._artifact_path)

    def _load(self, path: str):
        p = Path(path)
        self._artifact = joblib.load(p)
        self._artifact_path = p

    @staticmethod
    def _get_model_coef(model, X_df):
        coef = model.coef_
        intercept = model.intercept_
        classes = model.classes_

        # sklearn returns (n_classes-1, n_features) for binary in some cases
        # Normalize to (n_classes, n_features) for consistent reporting:
        if coef.shape[0] == 1 and len(classes) == 2:
            coef = np.vstack([-coef[0], coef[0]])
            intercept = np.array([-intercept[0], intercept[0]])

        return _coef_table(list(X_df.columns), classes, coef, intercept)

    @staticmethod
    def _get_explanation(coef_df, explain_top_k):
        explanation_path = Path("explanation.html")
        explanation_path.write_text(
            _top_features_html(coef_df, top_k=int(explain_top_k)),
            encoding="utf-8",
        )
        return explanation_path
