from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

MATCH_ID_COLUMN = "match_id_hash"
TARGET_COLUMN = "radiant_win"

# These values are taken from a completed OpenDota match. They are not valid
# predictors for an earlier snapshot, even if they correlate strongly with the result.
UNSAFE_EXACT_COLUMNS = frozenset(
    {
        "duration",
        "radiant_score",
        "dire_score",
        "radiant_win",
        "tower_status_radiant",
        "tower_status_dire",
    }
)
UNSAFE_COLUMN_FRAGMENTS = (
    "damage_taken",
    "hero_damage",
    "hero_healing",
    "roshans_killed",
    "total_damage",
    "towers_killed",
)
SAFE_PLAYER_METADATA = frozenset({"hero_id", "hero_name", "pred_vict"})


class DataContractError(ValueError):
    """Input data violates the documented training-time contract."""


def _require_columns(frame: pd.DataFrame, required: set[str], *, source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"{source} is missing required columns: {missing}")


def audit_snapshot_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return columns known to contain outcome or post-snapshot information."""
    unsafe = []
    for column in frame.columns:
        normalized = str(column).lower()
        if normalized in UNSAFE_EXACT_COLUMNS or any(
            fragment in normalized for fragment in UNSAFE_COLUMN_FRAGMENTS
        ):
            unsafe.append(str(column))
    return tuple(sorted(unsafe))


def merge_feature_target_tables(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    match_id_column: str = MATCH_ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and align feature and target tables without silent row loss."""
    _require_columns(features, {match_id_column}, source="feature table")
    _require_columns(targets, {match_id_column, target_column}, source="target table")

    if features[match_id_column].isna().any() or targets[match_id_column].isna().any():
        raise DataContractError("match identifiers must not contain missing values")
    if features[match_id_column].duplicated().any():
        raise DataContractError("feature table contains duplicate match identifiers")
    if targets[match_id_column].duplicated().any():
        raise DataContractError("target table contains duplicate match identifiers")

    feature_ids = set(features[match_id_column])
    target_ids = set(targets[match_id_column])
    if feature_ids != target_ids:
        missing_targets = len(feature_ids - target_ids)
        orphan_targets = len(target_ids - feature_ids)
        raise DataContractError(
            "feature and target identifiers differ: "
            f"missing_targets={missing_targets}, orphan_targets={orphan_targets}"
        )

    aligned = features.merge(
        targets[[match_id_column, target_column]],
        on=match_id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    target = aligned.pop(target_column)
    if target.isna().any():
        raise DataContractError("target contains missing values")

    try:
        target = target.astype("int8")
    except (TypeError, ValueError) as exc:
        raise DataContractError("target must contain only binary 0/1 values") from exc

    labels = set(target.unique())
    if labels != {0, 1}:
        raise DataContractError(f"target must contain both binary classes, got {sorted(labels)}")

    unsafe = audit_snapshot_columns(aligned)
    if unsafe:
        raise DataContractError(f"post-match or outcome columns are forbidden: {list(unsafe)}")

    return aligned.reset_index(drop=True), target.reset_index(drop=True)


def load_training_data(
    features_path: str | Path,
    targets_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the two distributed CSV files and enforce their joint contract."""
    features_file = Path(features_path)
    targets_file = Path(targets_path)
    if not features_file.is_file():
        raise FileNotFoundError(f"feature file not found: {features_file}")
    if not targets_file.is_file():
        raise FileNotFoundError(f"target file not found: {targets_file}")

    return merge_feature_target_tables(
        pd.read_csv(features_file),
        pd.read_csv(targets_file),
    )


def _player_prefix(player_slot: Any) -> str | None:
    if isinstance(player_slot, bool) or not isinstance(player_slot, int):
        return None
    if 0 <= player_slot <= 4:
        return f"r{player_slot + 1}"
    if 128 <= player_slot <= 132:
        return f"d{player_slot - 127}"
    return None


def extract_pregame_player_metadata(
    matches: Iterable[Mapping[str, Any]],
    *,
    fields: Sequence[str] = ("hero_name",),
    match_id_column: str = MATCH_ID_COLUMN,
) -> pd.DataFrame:
    """Extract only explicitly allow-listed, pre-game player metadata."""
    requested = tuple(fields)
    forbidden = sorted(set(requested).difference(SAFE_PLAYER_METADATA))
    if forbidden:
        raise DataContractError(
            f"player fields are not proven snapshot-safe and cannot be attached: {forbidden}"
        )

    rows: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for match in matches:
        match_id = match.get(match_id_column)
        if match_id is None:
            raise DataContractError(f"JSONL match is missing {match_id_column}")
        if match_id in seen_ids:
            raise DataContractError(f"duplicate JSONL match identifier: {match_id}")
        seen_ids.add(match_id)

        row: dict[str, Any] = {match_id_column: match_id}
        players = match.get("players")
        if not isinstance(players, list):
            raise DataContractError(f"match {match_id} has no player list")
        for player in players:
            if not isinstance(player, Mapping):
                continue
            prefix = _player_prefix(player.get("player_slot"))
            if prefix is None:
                continue
            for field in requested:
                row[f"{prefix}_{field}"] = player.get(field)
        rows.append(row)

    return pd.DataFrame(rows)


def load_pregame_metadata_jsonl(
    path: str | Path,
    *,
    fields: Sequence[str] = ("hero_name",),
) -> pd.DataFrame:
    """Stream a JSONL file and retain only approved pre-game fields."""
    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    def records() -> Iterable[Mapping[str, Any]]:
        with jsonl_path.open(encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataContractError(
                        f"invalid JSON on line {line_number} of {jsonl_path}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise DataContractError(f"JSONL line {line_number} is not an object")
                yield record

    return extract_pregame_player_metadata(records(), fields=fields)


def attach_pregame_metadata(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    match_id_column: str = MATCH_ID_COLUMN,
) -> pd.DataFrame:
    """Left-join safe metadata without overwriting snapshot columns."""
    _require_columns(features, {match_id_column}, source="feature table")
    _require_columns(metadata, {match_id_column}, source="metadata table")
    if metadata[match_id_column].duplicated().any():
        raise DataContractError("metadata table contains duplicate match identifiers")

    overlapping = sorted(
        set(features.columns).intersection(metadata.columns).difference({match_id_column})
    )
    if overlapping:
        raise DataContractError(f"metadata would overwrite existing columns: {overlapping}")

    return features.merge(
        metadata,
        on=match_id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
