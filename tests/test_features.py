from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dota_predictor.data import DataContractError
from dota_predictor.features import HeroWinRateEncoder, add_team_aggregates


def test_team_aggregates_are_symmetric_and_non_mutating() -> None:
    frame = pd.DataFrame(
        {
            **{f"r{slot}_gold": [slot * 10] for slot in range(1, 6)},
            **{f"d{slot}_gold": [slot * 5] for slot in range(1, 6)},
        }
    )
    result = add_team_aggregates(frame, metrics=("gold",))

    assert result.loc[0, "radiant_total_gold"] == 150
    assert result.loc[0, "dire_total_gold"] == 75
    assert result.loc[0, "radiant_gold_advantage"] == 75
    assert "radiant_total_gold" not in frame


def test_team_aggregates_reject_partial_metric() -> None:
    with pytest.raises(DataContractError, match="partial player metric"):
        add_team_aggregates(pd.DataFrame({"r1_gold": [100]}), metrics=("gold",))


def _unique_drafts() -> pd.DataFrame:
    rows = []
    for row_index in range(2):
        row = {}
        for slot in range(1, 6):
            row[f"r{slot}_hero_name"] = f"row{row_index}_r{slot}"
            row[f"d{slot}_hero_name"] = f"row{row_index}_d{slot}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_training_encoding_excludes_own_match_outcome() -> None:
    frame = _unique_drafts()
    encoder = HeroWinRateEncoder(smoothing=10)

    encoded = encoder.fit_transform(frame, pd.Series([0, 1]))
    full_mapping = encoder.transform(frame)

    encoded_columns = [
        column
        for column in encoded
        if column[0] in {"r", "d"} and column[1].isdigit() and column.endswith("_hero_winrate")
    ]
    assert np.allclose(encoded[encoded_columns], 0.5)
    assert not np.allclose(full_mapping[encoded_columns], 0.5)
    assert all(column not in encoded for column in frame.columns)


def test_unknown_hero_uses_training_prior() -> None:
    frame = _unique_drafts()
    encoder = HeroWinRateEncoder(smoothing=10).fit(frame, pd.Series([0, 1]))
    unknown = frame.iloc[[0]].copy()
    unknown.loc[:, "r1_hero_name"] = "brand_new_hero"

    transformed = encoder.transform(unknown)

    assert transformed.loc[0, "r1_hero_winrate"] == pytest.approx(encoder.prior_)


def test_encoder_rejects_duplicate_hero_in_draft() -> None:
    frame = _unique_drafts()
    frame.loc[0, "d1_hero_name"] = frame.loc[0, "r1_hero_name"]
    with pytest.raises(DataContractError, match="appears twice"):
        HeroWinRateEncoder().fit(frame, pd.Series([0, 1]))


def test_encoder_requires_dataframe_and_positive_smoothing() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        HeroWinRateEncoder().fit([["axe"]], [0])
    with pytest.raises(ValueError, match="greater than zero"):
        HeroWinRateEncoder(smoothing=0).fit(_unique_drafts(), [0, 1])


def test_team_aggregates_reject_non_numeric_values() -> None:
    frame = pd.DataFrame(
        {
            **{f"r{slot}_gold": ["unknown" if slot == 3 else 100] for slot in range(1, 6)},
            **{f"d{slot}_gold": [100] for slot in range(1, 6)},
        }
    )
    with pytest.raises(DataContractError, match="non-numeric"):
        add_team_aggregates(frame, metrics=("gold",))


def test_encoder_validates_column_and_target_contracts() -> None:
    frame = _unique_drafts()
    with pytest.raises(DataContractError, match="hero columns are missing"):
        HeroWinRateEncoder(hero_columns=("r1_hero_name", "missing_hero_name")).fit(frame, [0, 1])
    with pytest.raises(DataContractError, match="invalid hero column"):
        HeroWinRateEncoder(hero_columns=("r1_hero_name", "custom_hero_name")).fit(
            frame.assign(custom_hero_name=["a", "b"]), [0, 1]
        )
    with pytest.raises(DataContractError, match="no r1-r5"):
        HeroWinRateEncoder().fit(pd.DataFrame({"game_time": [1, 2]}), [0, 1])
    with pytest.raises(DataContractError, match="different row counts"):
        HeroWinRateEncoder().fit(frame, [0])
    with pytest.raises(DataContractError, match="binary and non-null"):
        HeroWinRateEncoder().fit(frame, [0, 2])
    with pytest.raises(DataContractError, match="target is required"):
        HeroWinRateEncoder().fit_transform(frame)


def test_encoder_validates_transform_contract() -> None:
    frame = _unique_drafts()
    encoder = HeroWinRateEncoder().fit(frame, [0, 1])
    with pytest.raises(TypeError, match="DataFrame"):
        encoder.transform(frame.to_numpy())
    with pytest.raises(DataContractError, match="missing at transform"):
        encoder.transform(frame.drop(columns="r1_hero_name"))
