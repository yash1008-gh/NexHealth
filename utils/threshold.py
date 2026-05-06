from __future__ import annotations

import json
from typing import Any

from src.predict import THRESHOLD_ARTIFACT_PATH


def save_threshold(threshold: float, metadata: dict | None = None) -> None:
    payload: dict[str, Any] = dict(metadata or {})
    payload["threshold"] = float(threshold)
    THRESHOLD_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_ARTIFACT_PATH.open("w", encoding="utf-8") as threshold_file:
        json.dump(payload, threshold_file, indent=2)


def load_threshold() -> float:
    metadata = load_threshold_metadata()
    if "threshold" not in metadata:
        raise ValueError(f"Threshold artifact is missing the 'threshold' key: {THRESHOLD_ARTIFACT_PATH}")
    return float(metadata["threshold"])


def load_threshold_metadata() -> dict:
    with THRESHOLD_ARTIFACT_PATH.open("r", encoding="utf-8") as threshold_file:
        return json.load(threshold_file)
