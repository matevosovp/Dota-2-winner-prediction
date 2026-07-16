from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import DataContractError


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    roc_auc: float
    best_params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NestedCVResult:
    fold_results: tuple[FoldResult, ...]
    oof_roc_auc: float
    mean_fold_roc_auc: float
    std_fold_roc_auc: float
    oof_predictions: np.ndarray

    def to_dict(self, *, include_predictions: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": "nested_stratified_cross_validation",
            "oof_roc_auc": self.oof_roc_auc,
            "mean_fold_roc_auc": self.mean_fold_roc_auc,
            "std_fold_roc_auc": self.std_fold_roc_auc,
            "folds": [asdict(result) for result in self.fold_results],
        }
        if include_predictions:
            payload["oof_predictions"] = self.oof_predictions.tolist()
        return payload


def _validated_target(y: Any, *, minimum_per_class: int) -> pd.Series:
    target = pd.Series(y).reset_index(drop=True)
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise DataContractError("nested CV requires a non-null target with both 0 and 1")
    counts = target.value_counts()
    if counts.min() < minimum_per_class:
        raise DataContractError(
            "not enough examples per class for the requested nested CV: "
            f"minimum={minimum_per_class}, counts={counts.to_dict()}"
        )
    return target.astype("int8")


def _minimum_class_count(outer_splits: int, inner_splits: int) -> int:
    """Smallest class size that leaves enough rows in every outer-train fold."""
    class_count = max(outer_splits, inner_splits)
    while class_count - math.ceil(class_count / outer_splits) < inner_splits:
        class_count += 1
    return class_count


def nested_cross_validate(
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    X: pd.DataFrame,
    y: Any,
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    n_jobs: int | None = -1,
) -> NestedCVResult:
    """Tune only on inner folds and report performance only on outer folds."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("nested_cross_validate requires a pandas DataFrame")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2")
    if len(X) == 0:
        raise DataContractError("cannot evaluate an empty dataset")

    target = _validated_target(
        y,
        minimum_per_class=_minimum_class_count(outer_splits, inner_splits),
    )
    if len(target) != len(X):
        raise DataContractError("features and target have different row counts")

    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof = np.full(len(X), np.nan, dtype=float)
    fold_results: list[FoldResult] = []

    for fold, (train_indices, validation_indices) in enumerate(
        outer_cv.split(X, target),
        start=1,
    ):
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state + fold,
        )
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(X.iloc[train_indices], target.iloc[train_indices])
        predictions = search.predict_proba(X.iloc[validation_indices])[:, 1]
        oof[validation_indices] = predictions
        fold_auc = float(roc_auc_score(target.iloc[validation_indices], predictions))
        fold_results.append(
            FoldResult(
                fold=fold,
                train_rows=len(train_indices),
                validation_rows=len(validation_indices),
                roc_auc=fold_auc,
                best_params=dict(search.best_params_),
            )
        )

    if np.isnan(oof).any():
        raise RuntimeError("nested CV did not produce exactly one prediction per row")
    fold_scores = np.asarray([result.roc_auc for result in fold_results])
    return NestedCVResult(
        fold_results=tuple(fold_results),
        oof_roc_auc=float(roc_auc_score(target, oof)),
        mean_fold_roc_auc=float(fold_scores.mean()),
        std_fold_roc_auc=float(fold_scores.std(ddof=0)),
        oof_predictions=oof,
    )
