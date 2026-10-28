from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig


@dataclass
class LogisticRegressionAppConfig(AppConfig):
    target_column: str = "target"
    standardize: bool = 1

    C: float = 1.0
    max_iter: int = 200
    solver: str = "lbfgs"  # lbfgs | liblinear | saga | newton-cg | sag
    class_weight: str = "balanced"  # balanced | none
    random_state: int = 42
    explain_top_k: int = 15  # top +/- features per class


@dataclass
class LogisticRegressionAppInputConfig(AppInputConfig):
    data: Any = None


@dataclass
class LogisticRegressionAppOutputConfig(AppOutputConfig):
    predictions: Any | None = None
    coefficients: Any | None = None
    explanation: Path | None = None
    report: str | None = None
