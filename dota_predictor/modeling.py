from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import MATCH_ID_COLUMN
from .features import HeroWinRateEncoder

RANDOM_STATE = 42


def _numeric_feature_columns(frame: object) -> list[str]:
    if not hasattr(frame, "select_dtypes"):
        raise TypeError("numeric preprocessing requires a pandas DataFrame")
    numeric = frame.select_dtypes(include=np.number).columns
    return [column for column in numeric if column != MATCH_ID_COLUMN]


def _numeric_preprocessor(*, scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [("numeric", Pipeline(steps), _numeric_feature_columns)],
        remainder="drop",
    )


def build_logistic_pipeline(
    *,
    smoothing: float = 20.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create an interpretable baseline with fold-local target encoding."""
    return Pipeline(
        [
            ("hero_winrate", HeroWinRateEncoder(smoothing=smoothing)),
            ("preprocess", _numeric_preprocessor(scale=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=3_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_catboost_pipeline(
    *,
    smoothing: float = 20.0,
    random_state: int = RANDOM_STATE,
    iterations: int = 600,
) -> Pipeline:
    """Create the tree candidate without reusing an outer validation fold."""
    return Pipeline(
        [
            ("hero_winrate", HeroWinRateEncoder(smoothing=smoothing)),
            ("preprocess", _numeric_preprocessor(scale=False)),
            (
                "model",
                CatBoostClassifier(
                    iterations=iterations,
                    depth=7,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                    # GridSearchCV parallelizes candidates; one thread per model avoids
                    # multiplying CPU usage across nested workers.
                    thread_count=1,
                ),
            ),
        ]
    )
