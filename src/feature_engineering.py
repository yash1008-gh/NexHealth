from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import (
    BINARY_FLAG_MAPS,
    DESCRIPTION_COLUMN_MAP,
    GROUP_COLUMN,
    ID_COLUMNS,
    LAB_RESULT_ORDINAL_MAPS,
    MEDICATION_COLUMNS,
    TRACEABILITY_ONLY_COLUMNS,
    deterministic_range_midpoints,
    normalize_missing_markers,
)


def map_icd9_group(code: object) -> str:
    if pd.isna(code):
        return "Unknown"

    text = str(code).strip()
    if not text:
        return "Unknown"
    if text.startswith(("V", "E")):
        return "Other"

    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if value == 250:
        return "Diabetes"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


@dataclass
class NexHealthFeatureEngineer(BaseEstimator, TransformerMixin):
    top_n_specialties: int = 10
    specialty_fill_value: str = "Unknown"
    learned_specialties_: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NexHealthFeatureEngineer":
        frame = self._prepare_frame(X)
        specialties = frame.get("medical_specialty", pd.Series(dtype="object")).fillna(self.specialty_fill_value)
        top_specialties = specialties.value_counts().nlargest(self.top_n_specialties).index.tolist()
        self.learned_specialties_ = top_specialties
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_frame(X)

        for diagnosis_column in ["diag_1", "diag_2", "diag_3"]:
            frame[f"{diagnosis_column}_group"] = frame[diagnosis_column].apply(map_icd9_group)
        frame = frame.drop(columns=["diag_1", "diag_2", "diag_3"], errors="ignore")

        frame["medical_specialty"] = frame["medical_specialty"].fillna(self.specialty_fill_value)
        if self.learned_specialties_:
            frame["medical_specialty"] = frame["medical_specialty"].where(
                frame["medical_specialty"].isin(self.learned_specialties_),
                "Other",
            )

        frame["service_utilization"] = (
            frame["number_outpatient"] + frame["number_emergency"] + frame["number_inpatient"]
        )
        frame["num_active_meds"] = frame[MEDICATION_COLUMNS].ne("No").sum(axis=1)
        frame["num_med_changes"] = frame[MEDICATION_COLUMNS].isin(["Up", "Down"]).sum(axis=1)
        frame["insulin_used"] = frame["insulin"].ne("No").astype(int)
        frame["insulin_change"] = frame["insulin"].isin(["Up", "Down"]).astype(int)
        frame["med_intensity"] = frame["num_medications"] / (frame["time_in_hospital"] + 1.0)
        frame["care_intensity"] = frame["num_lab_procedures"] / (frame["time_in_hospital"] + 1.0)
        frame["chronic_burden"] = frame["number_diagnoses"] * (frame["number_inpatient"] + 1)
        frame["relapse_probability"] = frame["number_inpatient"] * frame["number_diagnoses"]
        frame["is_high_comorbidity"] = (frame["number_diagnoses"] > 5).astype(int)
        frame["age_utilization"] = frame["age"] * frame["service_utilization"]

        for column, mapping in LAB_RESULT_ORDINAL_MAPS.items():
            frame[column] = frame[column].fillna("NotMeasured").map(mapping).astype(float)
        for column, mapping in BINARY_FLAG_MAPS.items():
            frame[column] = frame[column].fillna("No").map(mapping).astype(int)

        for description_column in DESCRIPTION_COLUMN_MAP.values():
            frame[description_column] = frame[description_column].fillna("Unknown")

        frame = frame.drop(columns=ID_COLUMNS + TRACEABILITY_ONLY_COLUMNS, errors="ignore")

        return frame

    def _prepare_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame = normalize_missing_markers(frame)
        frame = deterministic_range_midpoints(frame)
        return frame
