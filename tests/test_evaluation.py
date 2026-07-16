from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dota_predictor.data import DataContractError
from dota_predictor.evaluation import nested_cross_validate
from dota_predictor.features import add_team_aggregates
from dota_predictor.modeling import build_catboost_pipeline, build_logistic_pipeline


def test_nested_cv_produces_one_unbiased_oof_prediction_per_row(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = synthetic_training_data
    features = add_team_aggregates(features, metrics=("gold",))

    result = nested_cross_validate(
        build_logistic_pipeline(),
        {"model__C": [0.1, 1.0]},
        features,
        target,
        outer_splits=3,
        inner_splits=2,
        n_jobs=1,
    )

    assert len(result.fold_results) == 3
    assert result.oof_predictions.shape == (len(features),)
    assert np.isfinite(result.oof_predictions).all()
    assert 0.0 <= result.oof_roc_auc <= 1.0
    payload = result.to_dict(include_predictions=True)
    assert payload["protocol"] == "nested_stratified_cross_validation"
    assert len(payload["oof_predictions"]) == len(features)


def test_match_identifier_is_not_used_as_numeric_feature(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = synthetic_training_data
    features["match_id_hash"] = np.arange(len(features))
    pipeline = build_logistic_pipeline().fit(features, target)

    selected = pipeline.named_steps["preprocess"].transformers_[0][2]
    assert "match_id_hash" not in selected


def test_nested_cv_validates_inputs(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = synthetic_training_data
    estimator = build_logistic_pipeline()
    with pytest.raises(ValueError, match="at least 2"):
        nested_cross_validate(estimator, {}, features, target, outer_splits=1)
    with pytest.raises(DataContractError, match="both 0 and 1"):
        nested_cross_validate(estimator, {}, features.iloc[:10], pd.Series([1] * 10))
    with pytest.raises(TypeError, match="DataFrame"):
        nested_cross_validate(estimator, {}, features.to_numpy(), target)
    with pytest.raises(DataContractError, match="not enough examples"):
        nested_cross_validate(
            estimator,
            {},
            features.iloc[:6],
            pd.Series([0, 0, 0, 1, 1, 1]),
            outer_splits=5,
            inner_splits=3,
        )
    with pytest.raises(DataContractError, match="not enough examples"):
        nested_cross_validate(
            estimator,
            {},
            features.iloc[:8],
            pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
            outer_splits=3,
            inner_splits=3,
        )
    with pytest.raises(DataContractError, match="empty dataset"):
        nested_cross_validate(
            estimator,
            {},
            features.iloc[:0],
            target.iloc[:0],
            outer_splits=2,
            inner_splits=2,
        )
    with pytest.raises(DataContractError, match="different row counts"):
        nested_cross_validate(
            estimator,
            {},
            features,
            pd.concat([target, pd.Series([0])], ignore_index=True),
            outer_splits=2,
            inner_splits=2,
        )


def test_catboost_candidate_is_configured_without_training_side_effects() -> None:
    pipeline = build_catboost_pipeline(iterations=5)
    model = pipeline.named_steps["model"]
    assert model.get_param("allow_writing_files") is False
    assert model.get_param("iterations") == 5
    assert model.get_param("thread_count") == 1


def test_numeric_selector_requires_dataframe() -> None:
    from dota_predictor.modeling import _numeric_feature_columns

    with pytest.raises(TypeError, match="DataFrame"):
        _numeric_feature_columns(np.asarray([[1, 2]]))
