from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pathlib
import platform
from .preprocess import PROJECT_ROOT

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CALIBRATED_PIPELINE_ARTIFACT_PATH = ARTIFACTS_DIR / "calibrated_pipeline.pkl"
EXPLAINER_PIPELINE_ARTIFACT_PATH = ARTIFACTS_DIR / "explainer_pipeline.pkl"
PREPROCESSOR_ARTIFACT_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
SHAP_VALUES_ARTIFACT_PATH = ARTIFACTS_DIR / "shap_values.npy"
SHAP_SUMMARY_ARTIFACT_PATH = ARTIFACTS_DIR / "shap_summary.png"
THRESHOLD_ARTIFACT_PATH = ARTIFACTS_DIR / "threshold.json"

# Backward-compatible alias for the main inference artifact.
MODEL_ARTIFACT_PATH = CALIBRATED_PIPELINE_ARTIFACT_PATH


def save_pickle_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as artifact_file:
        pickle.dump(obj, artifact_file)


def load_pickle_artifact(path: Path) -> Any:
    if platform.system() == 'Linux':
        pathlib.WindowsPath = pathlib.PosixPath
    with path.open("rb") as artifact_file:
        return pickle.load(artifact_file)


def load_calibrated_pipeline():
    return load_pickle_artifact(CALIBRATED_PIPELINE_ARTIFACT_PATH)


def load_explainer_pipeline():
    return load_pickle_artifact(EXPLAINER_PIPELINE_ARTIFACT_PATH)


def load_preprocessor():
    return load_pickle_artifact(PREPROCESSOR_ARTIFACT_PATH)


def load_model():
    return load_calibrated_pipeline()
