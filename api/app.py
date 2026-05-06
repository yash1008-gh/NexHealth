from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import shap

from api.schemas import PatientFeatures, PredictionResponse, ShapContribution
from src.predict import load_calibrated_pipeline, load_explainer_pipeline
from utils.logger import get_app_logger
from utils.threshold import load_threshold, load_threshold_metadata

LOGGER = get_app_logger()
app = FastAPI(title="NexHealth Serving API")


@lru_cache(maxsize=1)
def get_calibrated_pipeline():
    return load_calibrated_pipeline()


@lru_cache(maxsize=1)
def get_explainer_pipeline():
    return load_explainer_pipeline()


@lru_cache(maxsize=1)
def get_threshold_value() -> float:
    return load_threshold()


@lru_cache(maxsize=1)
def get_threshold_info() -> dict:
    return load_threshold_metadata()


@app.on_event("startup")
def startup_event() -> None:
    get_calibrated_pipeline()
    get_explainer_pipeline()
    get_threshold_value()
    LOGGER.info("Serving artifacts loaded successfully on startup.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    LOGGER.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _normalize_local_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1])
    else:
        shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[..., -1]
    if shap_array.ndim == 2:
        return shap_array[0]
    return shap_array


def _build_local_shap_response(raw_row: pd.DataFrame) -> list[ShapContribution]:
    explainer_pipeline = get_explainer_pipeline()
    transformed = explainer_pipeline[:-1].transform(raw_row)
    transformed_frame = transformed if isinstance(transformed, pd.DataFrame) else pd.DataFrame(transformed)
    raw_model = explainer_pipeline.named_steps["model"]
    tree_model = getattr(raw_model, "booster_", raw_model)
    tree_explainer = shap.TreeExplainer(tree_model)
    local_shap = _normalize_local_shap_values(tree_explainer.shap_values(transformed_frame))

    contributions = (
        pd.Series(local_shap, index=transformed_frame.columns)
        .reindex(pd.Series(local_shap, index=transformed_frame.columns).abs().sort_values(ascending=False).index)
        .head(10)
    )
    return [
        ShapContribution(feature=str(feature), impact=float(impact))
        for feature, impact in contributions.items()
    ]


@app.get("/health")
def health() -> dict[str, object]:
    threshold_metadata = get_threshold_info()
    return {
        "status": "ok",
        "artifacts_loaded": True,
        "threshold_available": "threshold" in threshold_metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures, explain: bool = Query(default=False)) -> PredictionResponse:
    raw_payload = payload.model_dump(mode="python", by_alias=True)
    raw_row = pd.DataFrame([raw_payload])

    patient_number = raw_payload.get("patient_number")
    encounter_id = raw_payload.get("encounter_id")
    threshold = get_threshold_value()
    calibrated_pipeline = get_calibrated_pipeline()

    LOGGER.info(
        "Prediction request received. patient_number=%s encounter_id=%s explain=%s",
        patient_number,
        encounter_id,
        explain,
    )

    probability = float(calibrated_pipeline.predict_proba(raw_row)[0, 1])
    prediction = int(probability >= threshold)

    shap_values: list[ShapContribution] | None = None
    if explain:
        try:
            shap_values = _build_local_shap_response(raw_row)
        except Exception as exc:
            LOGGER.exception(
                "Local SHAP generation failed. patient_number=%s encounter_id=%s",
                patient_number,
                encounter_id,
            )
            raise HTTPException(status_code=500, detail="Local SHAP explanation generation failed.") from exc

    LOGGER.info(
        "Prediction complete. patient_number=%s encounter_id=%s probability=%.6f prediction=%s threshold=%.6f",
        patient_number,
        encounter_id,
        probability,
        prediction,
        threshold,
    )
    return PredictionResponse(
        probability=probability,
        prediction=prediction,
        threshold_used=float(threshold),
        shap_values=shap_values,
    )
