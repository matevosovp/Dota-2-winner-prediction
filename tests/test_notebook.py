from __future__ import annotations

import json
from pathlib import Path


def test_review_notebook_is_clean_and_compilable() -> None:
    notebook = json.loads(Path("Dota2winnerPrediction.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        compile(source, f"Dota2winnerPrediction.ipynb:code-cell-{index}", "exec")
