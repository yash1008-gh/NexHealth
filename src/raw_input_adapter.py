from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import DEFAULT_IDS_MAPPING_PATH, DESCRIPTION_COLUMN_MAP, load_ids_mapping


class RawInputAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, ids_mapping_path: Path | str = DEFAULT_IDS_MAPPING_PATH) -> None:
        self.ids_mapping_path = ids_mapping_path

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RawInputAdapter":
        self.mapping_frames_ = load_ids_mapping(self.ids_mapping_path)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mapping_frames_"):
            self.mapping_frames_ = load_ids_mapping(self.ids_mapping_path)

        if not isinstance(X, pd.DataFrame):
            raise TypeError("RawInputAdapter expects a pandas DataFrame as input.")

        transformed = X.copy()
        original_row_count = len(transformed)

        for id_column, description_column in DESCRIPTION_COLUMN_MAP.items():
            if description_column in transformed.columns:
                continue
            if id_column not in transformed.columns:
                raise ValueError(
                    f"Cannot create '{description_column}' because required raw ID column '{id_column}' is missing."
                )

            mapping_frame = self.mapping_frames_[id_column][[id_column, description_column]].copy()
            transformed = transformed.merge(mapping_frame, on=id_column, how="left", validate="many_to_one")
            transformed[description_column] = transformed[description_column].fillna("Unknown")

            suffix_columns = [column for column in transformed.columns if column.endswith("_x") or column.endswith("_y")]
            if suffix_columns:
                raise ValueError(
                    "RawInputAdapter created unexpected suffix columns during merge: "
                    + ", ".join(suffix_columns)
                )

        if len(transformed) != original_row_count:
            raise ValueError(
                f"RawInputAdapter changed row count from {original_row_count} to {len(transformed)}."
            )

        if transformed.columns.duplicated().any():
            duplicate_columns = transformed.columns[transformed.columns.duplicated()].tolist()
            raise ValueError(
                "RawInputAdapter created duplicate columns: " + ", ".join(duplicate_columns)
            )

        return transformed
