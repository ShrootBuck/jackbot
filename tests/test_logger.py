from __future__ import annotations

import math
from pathlib import Path

from jackbot.training.logger import _json_safe_metadata


def test_json_safe_metadata_replaces_nonfinite_floats() -> None:
    metadata = _json_safe_metadata(
        {
            "best_score": float("-inf"),
            "nested": [1.0, float("inf"), math.nan],
            "path": Path("checkpoints/jackbot_latest.pt"),
        }
    )

    assert metadata == {
        "best_score": None,
        "nested": [1.0, None, None],
        "path": "checkpoints/jackbot_latest.pt",
    }
