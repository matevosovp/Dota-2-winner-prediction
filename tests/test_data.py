from __future__ import annotations

import json

import pandas as pd
import pytest

from dota_predictor.data import (
    DataContractError,
    attach_pregame_metadata,
    audit_snapshot_columns,
    extract_pregame_player_metadata,
    load_pregame_metadata_jsonl,
    load_training_data,
    merge_feature_target_tables,
)


def test_merge_aligns_targets_by_match_id() -> None:
    features = pd.DataFrame({"match_id_hash": ["a", "b"], "game_time": [600, 700]})
    targets = pd.DataFrame({"match_id_hash": ["b", "a"], "radiant_win": [1, 0]})

    aligned, target = merge_feature_target_tables(features, targets)

    assert aligned["match_id_hash"].tolist() == ["a", "b"]
    assert target.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("features", "targets", "message"),
    [
        (
            pd.DataFrame({"match_id_hash": ["a", "a"]}),
            pd.DataFrame({"match_id_hash": ["a", "b"], "radiant_win": [0, 1]}),
            "duplicate match",
        ),
        (
            pd.DataFrame({"match_id_hash": ["a", "b"]}),
            pd.DataFrame({"match_id_hash": ["a", "c"], "radiant_win": [0, 1]}),
            "identifiers differ",
        ),
        (
            pd.DataFrame({"match_id_hash": ["a", "b"]}),
            pd.DataFrame({"match_id_hash": ["a", "b"], "radiant_win": [1, 1]}),
            "both binary classes",
        ),
    ],
)
def test_merge_rejects_broken_contract(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(DataContractError, match=message):
        merge_feature_target_tables(features, targets)


def test_snapshot_audit_detects_post_match_features() -> None:
    frame = pd.DataFrame(
        {
            "game_time": [600],
            "r1_gold": [1_000],
            "r1_hero_damage": [9_999],
            "dire_score": [20],
        }
    )
    assert audit_snapshot_columns(frame) == ("dire_score", "r1_hero_damage")


def test_extracts_only_allowlisted_player_metadata() -> None:
    matches = [
        {
            "match_id_hash": "a",
            "players": [
                {"player_slot": 0, "hero_name": "npc_dota_hero_axe", "damage": 99},
                {"player_slot": 128, "hero_name": "npc_dota_hero_lina", "damage": 100},
            ],
        }
    ]

    metadata = extract_pregame_player_metadata(matches, fields=("hero_name",))

    assert metadata.to_dict(orient="records") == [
        {
            "match_id_hash": "a",
            "r1_hero_name": "npc_dota_hero_axe",
            "d1_hero_name": "npc_dota_hero_lina",
        }
    ]
    with pytest.raises(DataContractError, match="not proven snapshot-safe"):
        extract_pregame_player_metadata(matches, fields=("damage",))


def test_jsonl_loader_and_metadata_join(tmp_path) -> None:
    jsonl = tmp_path / "matches.jsonl"
    record = {
        "match_id_hash": "a",
        "players": [{"player_slot": 0, "hero_name": "npc_dota_hero_axe"}],
    }
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    metadata = load_pregame_metadata_jsonl(jsonl)
    joined = attach_pregame_metadata(
        pd.DataFrame({"match_id_hash": ["a", "b"], "game_time": [600, 700]}),
        metadata,
    )

    assert joined.loc[0, "r1_hero_name"] == "npc_dota_hero_axe"
    assert pd.isna(joined.loc[1, "r1_hero_name"])


def test_load_training_data_checks_files_and_columns(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="feature file"):
        load_training_data(tmp_path / "missing.csv", tmp_path / "targets.csv")

    features_path = tmp_path / "features.csv"
    targets_path = tmp_path / "targets.csv"
    pd.DataFrame({"match_id_hash": ["a", "b"], "game_time": [1, 2]}).to_csv(
        features_path, index=False
    )
    pd.DataFrame({"match_id_hash": ["a", "b"], "radiant_win": [0, 1]}).to_csv(
        targets_path, index=False
    )
    features, target = load_training_data(features_path, targets_path)
    assert len(features) == len(target) == 2


def test_metadata_join_refuses_overwrite() -> None:
    features = pd.DataFrame({"match_id_hash": ["a"], "r1_hero_name": ["axe"]})
    metadata = pd.DataFrame({"match_id_hash": ["a"], "r1_hero_name": ["lina"]})
    with pytest.raises(DataContractError, match="overwrite"):
        attach_pregame_metadata(features, metadata)


def test_merge_rejects_missing_columns_null_ids_and_unsafe_features() -> None:
    valid_targets = pd.DataFrame({"match_id_hash": ["a", "b"], "radiant_win": [0, 1]})
    with pytest.raises(DataContractError, match="missing required columns"):
        merge_feature_target_tables(pd.DataFrame({"game_time": [1, 2]}), valid_targets)
    with pytest.raises(DataContractError, match="missing values"):
        merge_feature_target_tables(
            pd.DataFrame({"match_id_hash": ["a", None]}),
            pd.DataFrame({"match_id_hash": ["a", "b"], "radiant_win": [0, 1]}),
        )
    with pytest.raises(DataContractError, match="duplicate match"):
        merge_feature_target_tables(
            pd.DataFrame({"match_id_hash": ["a", "b"]}),
            pd.DataFrame({"match_id_hash": ["a", "a"], "radiant_win": [0, 1]}),
        )
    with pytest.raises(DataContractError, match="post-match"):
        merge_feature_target_tables(
            pd.DataFrame(
                {
                    "match_id_hash": ["a", "b"],
                    "r1_total_damage": [10, 20],
                }
            ),
            valid_targets,
        )
    with pytest.raises(DataContractError, match="binary 0/1"):
        merge_feature_target_tables(
            pd.DataFrame({"match_id_hash": ["a", "b"]}),
            pd.DataFrame({"match_id_hash": ["a", "b"], "radiant_win": [0, "no"]}),
        )


def test_metadata_contract_rejects_bad_records_and_duplicates() -> None:
    with pytest.raises(DataContractError, match="missing match_id_hash"):
        extract_pregame_player_metadata([{"players": []}])
    with pytest.raises(DataContractError, match="duplicate JSONL"):
        extract_pregame_player_metadata(
            [
                {"match_id_hash": "a", "players": []},
                {"match_id_hash": "a", "players": []},
            ]
        )
    with pytest.raises(DataContractError, match="no player list"):
        extract_pregame_player_metadata([{"match_id_hash": "a", "players": None}])

    metadata = extract_pregame_player_metadata(
        [
            {
                "match_id_hash": "a",
                "players": ["not-a-player", {"player_slot": True, "hero_name": "axe"}],
            }
        ]
    )
    assert metadata.columns.tolist() == ["match_id_hash"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{broken}\n", "invalid JSON"),
        ("[]\n", "is not an object"),
    ],
)
def test_jsonl_loader_rejects_malformed_records(tmp_path, content: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DataContractError, match=message):
        load_pregame_metadata_jsonl(path)


def test_metadata_join_rejects_duplicate_ids() -> None:
    features = pd.DataFrame({"match_id_hash": ["a"]})
    metadata = pd.DataFrame({"match_id_hash": ["a", "a"]})
    with pytest.raises(DataContractError, match="duplicate match"):
        attach_pregame_metadata(features, metadata)
