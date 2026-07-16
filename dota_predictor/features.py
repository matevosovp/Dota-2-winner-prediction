from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .data import DataContractError

HERO_COLUMN_PATTERN = re.compile(r"^[rd][1-5]_hero_name$")
DEFAULT_TEAM_METRICS = (
    "gold",
    "xp",
    "level",
    "kills",
    "deaths",
    "assists",
    "lh",
    "denies",
    "stuns",
)


def _team_columns(metric: str, side: str) -> list[str]:
    return [f"{side}{slot}_{metric}" for slot in range(1, 6)]


def add_team_aggregates(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_TEAM_METRICS,
) -> pd.DataFrame:
    """Add symmetric team totals and Radiant-minus-Dire advantages."""
    result = frame.copy()
    for metric in metrics:
        radiant_columns = _team_columns(metric, "r")
        dire_columns = _team_columns(metric, "d")
        expected = radiant_columns + dire_columns
        present = [column for column in expected if column in result.columns]
        if not present:
            continue
        if len(present) != len(expected):
            missing = sorted(set(expected).difference(present))
            raise DataContractError(f"partial player metric {metric!r}; missing columns: {missing}")

        numeric = result[expected].apply(pd.to_numeric, errors="coerce")
        invalid = result[expected].notna() & numeric.isna()
        if invalid.any().any():
            raise DataContractError(f"player metric {metric!r} contains non-numeric values")

        radiant_total = numeric[radiant_columns].sum(axis=1, min_count=1)
        dire_total = numeric[dire_columns].sum(axis=1, min_count=1)
        result[f"radiant_total_{metric}"] = radiant_total
        result[f"dire_total_{metric}"] = dire_total
        result[f"radiant_{metric}_advantage"] = radiant_total - dire_total
    return result


def _normalize_hero_names(series: pd.Series) -> pd.Series:
    return (
        series.astype("string").str.replace(r"^npc_dota_hero_", "", regex=True).replace("", pd.NA)
    )


class HeroWinRateEncoder(TransformerMixin, BaseEstimator):
    """Smoothed hero target encoder with leave-one-match-out training values.

    ``fit_transform`` is used on each training fold by sklearn Pipeline. It removes the
    current match's outcome from every encoded hero statistic. ``transform`` uses the
    mapping learned only from the corresponding training fold.
    """

    def __init__(
        self,
        hero_columns: tuple[str, ...] | None = None,
        *,
        smoothing: float = 20.0,
        drop_hero_names: bool = True,
        add_team_features: bool = True,
    ) -> None:
        self.hero_columns = hero_columns
        self.smoothing = smoothing
        self.drop_hero_names = drop_hero_names
        self.add_team_features = add_team_features

    def _resolve_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        columns = self.hero_columns
        if columns is None:
            columns = tuple(
                column for column in frame.columns if HERO_COLUMN_PATTERN.fullmatch(str(column))
            )
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise DataContractError(f"hero columns are missing: {missing}")
        if not columns:
            raise DataContractError("no r1-r5/d1-d5 hero name columns were found")
        invalid = [column for column in columns if not HERO_COLUMN_PATTERN.fullmatch(column)]
        if invalid:
            raise DataContractError(f"invalid hero column names: {invalid}")
        return tuple(columns)

    @staticmethod
    def _validate_target(target: Any, expected_length: int) -> pd.Series:
        values = pd.Series(target).reset_index(drop=True)
        if len(values) != expected_length:
            raise DataContractError("features and target have different row counts")
        if values.isna().any() or not set(values.unique()).issubset({0, 1}):
            raise DataContractError("hero encoder target must be binary and non-null")
        return values.astype(float)

    @staticmethod
    def _validate_unique_drafts(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        normalized = frame.loc[:, columns].apply(_normalize_hero_names)
        duplicated = normalized.apply(
            lambda row: row.dropna().duplicated().any(),
            axis=1,
        )
        if duplicated.any():
            rows = duplicated[duplicated].index.tolist()[:5]
            raise DataContractError(f"a hero appears twice in the same draft; rows={rows}")

    def _build_occurrences(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> pd.DataFrame:
        occurrences = []
        for column in self.hero_columns_:
            hero = _normalize_hero_names(frame[column]).reset_index(drop=True)
            hero_won = target if column.startswith("r") else 1.0 - target
            occurrences.append(pd.DataFrame({"hero": hero, "won": hero_won}))
        return pd.concat(occurrences, ignore_index=True).dropna(subset=["hero"])

    def fit(self, X: pd.DataFrame, y: Any) -> HeroWinRateEncoder:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeroWinRateEncoder requires a pandas DataFrame")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be greater than zero")

        self.hero_columns_ = self._resolve_columns(X)
        self._validate_unique_drafts(X, self.hero_columns_)
        target = self._validate_target(y, len(X))
        occurrences = self._build_occurrences(X.reset_index(drop=True), target)
        if occurrences.empty:
            raise DataContractError("hero columns contain no usable hero names")

        stats = occurrences.groupby("hero")["won"].agg(["sum", "count"])
        self.hero_wins_ = stats["sum"].astype(float)
        self.hero_counts_ = stats["count"].astype(float)
        self.prior_ = float(occurrences["won"].mean())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _encoded_column(
        self,
        heroes: pd.Series,
        *,
        row_wins: pd.Series | None = None,
    ) -> pd.Series:
        normalized = _normalize_hero_names(heroes)
        wins = normalized.map(self.hero_wins_)
        counts = normalized.map(self.hero_counts_)
        if row_wins is not None:
            wins = wins - row_wins.to_numpy()
            counts = counts - 1.0

        encoded = (wins + self.smoothing * self.prior_) / (counts + self.smoothing)
        return encoded.fillna(self.prior_).astype(float)

    def _transform(
        self,
        X: pd.DataFrame,
        *,
        target: pd.Series | None = None,
    ) -> pd.DataFrame:
        check_is_fitted(self, attributes=["hero_counts_", "hero_wins_", "prior_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeroWinRateEncoder requires a pandas DataFrame")
        missing = sorted(set(self.hero_columns_).difference(X.columns))
        if missing:
            raise DataContractError(f"hero columns are missing at transform time: {missing}")
        self._validate_unique_drafts(X, self.hero_columns_)

        result = X.copy()
        created: list[str] = []
        for column in self.hero_columns_:
            row_wins = None
            if target is not None:
                row_wins = target if column.startswith("r") else 1.0 - target
            encoded_name = column.replace("_hero_name", "_hero_winrate")
            result[encoded_name] = self._encoded_column(
                result[column], row_wins=row_wins
            ).to_numpy()
            created.append(encoded_name)

        if self.add_team_features:
            radiant = sorted(column for column in created if column.startswith("r"))
            dire = sorted(column for column in created if column.startswith("d"))
            result["radiant_total_hero_winrate"] = result[radiant].sum(axis=1)
            result["dire_total_hero_winrate"] = result[dire].sum(axis=1)
            result["radiant_hero_winrate_advantage"] = (
                result["radiant_total_hero_winrate"] - result["dire_total_hero_winrate"]
            )

        if self.drop_hero_names:
            result = result.drop(columns=list(self.hero_columns_))
        return result

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        del fit_params
        if y is None:
            raise DataContractError("target is required for leak-free hero encoding")
        self.fit(X, y)
        target = self._validate_target(y, len(X))
        return self._transform(X.reset_index(drop=True), target=target)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)
