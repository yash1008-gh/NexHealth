from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "diabetic_data.csv"
DEFAULT_IDS_MAPPING_PATH = DEFAULT_DATA_DIR / "IDS_mapping.csv"
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}
TARGET_COLUMN = "readmitted"
GROUP_COLUMN = "patient_number"
ID_COLUMNS = ["encounter_id", GROUP_COLUMN]
TRACEABILITY_ONLY_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]
DESCRIPTION_COLUMN_MAP = {
    "admission_type_id": "admission_type_desc",
    "discharge_disposition_id": "discharge_desc",
    "admission_source_id": "admission_source_desc",
}

LAB_RESULT_ORDINAL_MAPS = {
    "A1Cresult": {"NotMeasured": 0, "Norm": 1, ">7": 2, ">8": 3},
    "max_glu_serum": {"NotMeasured": 0, "Norm": 1, ">200": 2, ">300": 3},
}

BINARY_FLAG_MAPS = {
    "change": {"No": 0, "Ch": 1},
    "diabetesMed": {"No": 0, "Yes": 1},
}

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

def _range_midpoint(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text or text == "?":
        return np.nan

    range_match = re.match(r"^[\[\(]?\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[\)\]]?$", text)
    if range_match:
        start, end = map(float, range_match.groups())
        return (start + end) / 2.0

    greater_match = re.match(r"^>(\d+(?:\.\d+)?)$", text)    #
    if greater_match:
        return float(greater_match.group(1))

    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_missing_markers(df: pd.DataFrame, markers: Iterable[str] | None = None) -> pd.DataFrame:
    normalized = df.copy()
    marker_values = list(markers or ["?"])
    object_columns = normalized.select_dtypes(include=["object"]).columns
    for column in object_columns:
        normalized[column] = normalized[column].where(~normalized[column].isin(marker_values), np.nan)
    return normalized


def load_raw_data(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(data_path)
    df = pd.read_csv(path)
    if "patient_nbr" in df.columns:
        df = df.rename(columns={"patient_nbr": GROUP_COLUMN})
    return merge_description_columns(df)


def load_ids_mapping(ids_mapping_path: Path | str = DEFAULT_IDS_MAPPING_PATH) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(ids_mapping_path)

    key_col, desc_col = df.columns[:2]

    sections = {key: [] for key in DESCRIPTION_COLUMN_MAP}
    current_section = None

    for key, desc in zip(df[key_col], df[desc_col]):

        # Detect section headers
        if isinstance(key, str) and key in DESCRIPTION_COLUMN_MAP:
            current_section = key
            continue

        # Skip until a valid section is found
        if current_section is None:
            continue

        # Convert key safely
        try:
            numeric_key = int(float(key))
        except (ValueError, TypeError):
            continue

        sections[current_section].append({
            current_section: numeric_key,
            DESCRIPTION_COLUMN_MAP[current_section]: desc if pd.notna(desc) else "Unknown"
        })

    # Convert to DataFrames and enforce schema
    mapping_frames = {}
    for section, rows in sections.items():
        df_section = pd.DataFrame(rows)

        # Enforce expected columns even if empty
        expected_cols = [section, DESCRIPTION_COLUMN_MAP[section]]
        for col in expected_cols:
            if col not in df_section.columns:
                df_section[col] = []

        mapping_frames[section] = df_section[expected_cols]

    return mapping_frames


def merge_description_columns(
    df: pd.DataFrame,
    ids_mapping_path: Path | str = DEFAULT_IDS_MAPPING_PATH,
) -> pd.DataFrame:
    merged = df.copy()
    mapping_frames = load_ids_mapping(ids_mapping_path)

    for id_column, description_column in DESCRIPTION_COLUMN_MAP.items():
        mapping_frame = mapping_frames[id_column]
        merged = merged.merge(mapping_frame, on=id_column, how="left")
        merged[description_column] = merged[description_column].fillna("Unknown")

    return merged


def apply_eligibility_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[~filtered["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    if "gender" in filtered.columns:
        filtered = filtered[filtered["gender"] != "Unknown/Invalid"]
    return filtered.reset_index(drop=True)


def prepare_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    prepared = df.copy()
    prepared[target_column] = prepared[target_column].map({"NO": 0, ">30": 1, "<30": 1}).astype(int)
    return prepared


def deterministic_range_midpoints(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    if "age" in transformed.columns:
        transformed["age"] = transformed["age"].map(_range_midpoint)
    if "weight" in transformed.columns:
        transformed["weight"] = transformed["weight"].map(_range_midpoint)
    return transformed

import pandas as pd

def convert_to_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df

def build_preprocessor_ohe() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True, keep_empty_features=True)),
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),

        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )
    
    return preprocessor.set_output(transform="pandas")

def build_preprocessor_lgbm() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    ).set_output(transform="pandas")