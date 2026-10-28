from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


def _split_id_and_df(df: pd.DataFrame) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")
    if str(df.columns[0]).strip().lower() == "id":
        return df.iloc[:, 0], df.iloc[:, 1:]
    return None, df


def _pop_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def _ensure_numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    Xn = X.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if Xn.ndim != 2 or Xn.shape[1] == 0:
        raise ValueError("Feature matrix must be 2D with at least one feature column.")
    return Xn


def _write_predictions(
        ids: Optional[pd.Series],
        labels: np.ndarray,
        proba: np.ndarray,
        classes: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame({"label": labels})
    if ids is None:
        out.insert(0, "index", np.arange(len(labels)))
    else:
        out.insert(0, "id", ids.values)

    # Always output probabilities
    # Name columns by class label (stringified)
    for j, c in enumerate(classes):
        out[f"p_{str(c)}"] = proba[:, j]

    return out


def _coef_table(
        feature_names: List[str],
        classes: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
) -> pd.DataFrame:
    """
    Returns a long-form coefficient table:
      columns: class, feature, coef
    Also includes intercept as feature="(intercept)".
    """
    rows = []
    for ci, c in enumerate(classes):
        rows.append({"class": str(c), "feature": "(intercept)", "coef": float(intercept[ci])})
        for fi, fn in enumerate(feature_names):
            rows.append({"class": str(c), "feature": fn, "coef": float(coef[ci, fi])})
    return pd.DataFrame(rows)


def _top_features_html(coef_df: pd.DataFrame, top_k: int = 15) -> str:
    """
    Simple explainability HTML:
      For each class: show top positive and top negative coefficients.
    """
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;}",
        "table{border-collapse:collapse;width:100%;max-width:980px;margin-bottom:24px;}",
        "th,td{border:1px solid #ddd;padding:8px;}",
        "th{background:#f6f6f6;text-align:left;}",
        "code{background:#f1f1f1;padding:1px 4px;border-radius:4px;}",
        "</style></head><body>",
        "<h2>Logistic Regression Explainability</h2>",
        "<p>Coefficients are shown per class. Larger positive coefficients increase the log-odds of the class; negative decrease.</p>",
    ]

    for cls in sorted(coef_df["class"].unique()):
        sub = coef_df[(coef_df["class"] == cls) & (coef_df["feature"] != "(intercept)")].copy()
        sub = sub.sort_values("coef")
        neg = sub.head(top_k)
        pos = sub.tail(top_k).sort_values("coef", ascending=False)

        parts.append(f"<h3>Class: <code>{cls}</code></h3>")

        parts.append("<h4>Top positive coefficients</h4>")
        parts.append("<table><tr><th>Feature</th><th>Coef</th></tr>")
        for _, r in pos.iterrows():
            parts.append(f"<tr><td><code>{r['feature']}</code></td><td>{r['coef']:.6g}</td></tr>")
        parts.append("</table>")

        parts.append("<h4>Top negative coefficients</h4>")
        parts.append("<table><tr><th>Feature</th><th>Coef</th></tr>")
        for _, r in neg.iterrows():
            parts.append(f"<tr><td><code>{r['feature']}</code></td><td>{r['coef']:.6g}</td></tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)
