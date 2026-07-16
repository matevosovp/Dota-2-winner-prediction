from __future__ import annotations

import json

import pandas as pd

from dota_predictor.train import _model_and_grid, main


def test_cli_runs_nested_evaluation_and_writes_metrics(
    tmp_path,
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    capsys,
) -> None:
    features, target = synthetic_training_data
    hero_columns = [column for column in features if column.endswith("_hero_name")]
    features_path = tmp_path / "train_features.csv"
    targets_path = tmp_path / "train_targets.csv"
    matches_path = tmp_path / "train_matches.jsonl"
    output_path = tmp_path / "metrics" / "nested-cv.json"

    features.drop(columns=hero_columns).to_csv(features_path, index=False)
    pd.DataFrame(
        {
            "match_id_hash": features["match_id_hash"],
            "radiant_win": target,
        }
    ).to_csv(targets_path, index=False)

    with matches_path.open("w", encoding="utf-8") as file_obj:
        for _, row in features.iterrows():
            players = []
            for slot in range(1, 6):
                players.append(
                    {
                        "player_slot": slot - 1,
                        "hero_name": row[f"r{slot}_hero_name"],
                    }
                )
                players.append(
                    {
                        "player_slot": slot + 127,
                        "hero_name": row[f"d{slot}_hero_name"],
                    }
                )
            file_obj.write(
                json.dumps({"match_id_hash": row["match_id_hash"], "players": players}) + "\n"
            )

    exit_code = main(
        [
            "--features",
            str(features_path),
            "--targets",
            str(targets_path),
            "--matches-jsonl",
            str(matches_path),
            "--outer-splits",
            "2",
            "--inner-splits",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["model"] == "logistic"
    assert len(payload["folds"]) == 2
    assert '"protocol": "nested_stratified_cross_validation"' in capsys.readouterr().out


def test_cli_exposes_catboost_search_space() -> None:
    estimator, grid = _model_and_grid("catboost")
    assert estimator.named_steps["model"].get_param("allow_writing_files") is False
    assert grid["model__depth"] == [5, 7]
