from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import canonicalize_column_list, infer_numeric_and_categorical


DEFAULT_NON_FEATURE_COLUMNS = {
    "binary_label",
    "multiclass_label",
    "source_file",
    "row_id",
}


def select_feature_columns(
    df: pd.DataFrame,
    exclude_columns: Iterable[str] = (),
) -> list[str]:
    exclude = set(DEFAULT_NON_FEATURE_COLUMNS)
    exclude.update(canonicalize_column_list(exclude_columns))
    feature_cols = [col for col in df.columns if col not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns were selected. Check exclude_columns.")
    return feature_cols


def build_tabular_preprocessor(
    df: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_numeric_and_categorical(df, feature_columns)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
