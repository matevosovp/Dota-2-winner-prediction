from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import attach_pregame_metadata, load_pregame_metadata_jsonl, load_training_data
from .evaluation import nested_cross_validate
from .features import add_team_aggregates
from .modeling import build_catboost_pipeline, build_logistic_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leak-resistant nested-CV evaluation for Dota 2 match snapshots."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--matches-jsonl", type=Path, required=True)
    parser.add_argument("--model", choices=("logistic", "catboost"), default="logistic")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser


def _model_and_grid(name: str) -> tuple[Any, dict[str, list[Any]]]:
    if name == "catboost":
        return build_catboost_pipeline(), {
            "model__depth": [5, 7],
            "model__learning_rate": [0.03, 0.08],
        }
    return build_logistic_pipeline(), {
        "model__C": [0.1, 1.0, 10.0],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features, target = load_training_data(args.features, args.targets)
    metadata = load_pregame_metadata_jsonl(args.matches_jsonl, fields=("hero_name",))
    features = attach_pregame_metadata(features, metadata)
    features = add_team_aggregates(features)

    estimator, grid = _model_and_grid(args.model)
    result = nested_cross_validate(
        estimator,
        grid,
        features,
        target,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )
    payload = result.to_dict()
    payload["model"] = args.model
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
