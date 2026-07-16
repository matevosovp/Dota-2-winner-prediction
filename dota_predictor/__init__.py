"""Leak-resistant Dota 2 match outcome modeling utilities."""

from .data import DataContractError, load_training_data
from .evaluation import NestedCVResult, nested_cross_validate
from .features import HeroWinRateEncoder, add_team_aggregates
from .modeling import build_catboost_pipeline, build_logistic_pipeline

__all__ = [
    "DataContractError",
    "HeroWinRateEncoder",
    "NestedCVResult",
    "add_team_aggregates",
    "build_catboost_pipeline",
    "build_logistic_pipeline",
    "load_training_data",
    "nested_cross_validate",
]
