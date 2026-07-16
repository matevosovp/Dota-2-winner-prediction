from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    rows = 60
    target = pd.Series(np.tile([0, 1], rows // 2), dtype="int8")
    frame = pd.DataFrame(
        {
            "match_id_hash": [f"match-{index:03d}" for index in range(rows)],
            "game_time": rng.integers(300, 1_800, size=rows),
            "snapshot_signal": target.to_numpy() + rng.normal(0, 0.35, size=rows),
        }
    )

    hero_pool = [f"hero_{index}" for index in range(30)]
    for row_index in range(rows):
        draft = [hero_pool[(row_index + offset) % len(hero_pool)] for offset in range(10)]
        for slot in range(1, 6):
            frame.loc[row_index, f"r{slot}_hero_name"] = draft[slot - 1]
            frame.loc[row_index, f"d{slot}_hero_name"] = draft[slot + 4]
            frame.loc[row_index, f"r{slot}_gold"] = 1_000 + 50 * slot + int(target[row_index]) * 40
            frame.loc[row_index, f"d{slot}_gold"] = 1_000 + 50 * slot
    return frame, target
