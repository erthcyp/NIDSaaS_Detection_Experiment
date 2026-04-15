from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import canonicalize_columns, find_label_column, normalize_attack_label


CSV_GLOBS = ("*.csv")

def log(msg: str) -> None:
    print(f"[load_data] {msg}", flush=True)

@dataclass
class DetectionSplits:
    train_all: pd.DataFrame
    val_all: pd.DataFrame
    test_all: pd.DataFrame
    train_benign: pd.DataFrame
    val_benign: pd.DataFrame


def _iter_csv_paths(data_dir):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_dir}")

    seen = set()

    for path in sorted(data_dir.rglob("*")):
        # เอาเฉพาะไฟล์จริง
        if not path.is_file():
            continue

        # เอาเฉพาะ .csv
        if path.suffix.lower() != ".csv":
            continue

        key = str(path.resolve()).lower()
        if key in seen:
            continue

        seen.add(key)
        yield path


def read_cic_ids2017_folder(data_dir):
    data_dir = Path(data_dir)
    paths = list(_iter_csv_paths(data_dir))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")
    for i, p in enumerate(paths, start=1):
        log(f"[{i}] csv file = {p}")
    log(f"found {len(paths)} CSV files under: {data_dir}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        log(f"reading: {path}")
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df["source_file"] = path.name
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = canonicalize_columns(merged)
    merged["row_id"] = np.arange(len(merged), dtype=np.int64)
    log(f"merged rows={len(merged):,}, cols={merged.shape[1]:,}")
    return merged


def clean_detection_dataframe(
    df: pd.DataFrame,
    max_missing_fraction: float = 0.30,
    drop_unknown_labels: bool = True,
) -> pd.DataFrame:
    label_col = find_label_column(df)
    df = df.copy()
    log(f"raw rows before clean={len(df):,}")
    df[label_col] = df[label_col].map(normalize_attack_label)
    df = df.rename(columns={label_col: "multiclass_label"})
    if drop_unknown_labels:
        df = df.loc[df["multiclass_label"] != "UNKNOWN"].copy()
    df["binary_label"] = (df["multiclass_label"] != "BENIGN").astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)

    missing_fraction = df.isna().mean(axis=1)
    df = df.loc[missing_fraction <= max_missing_fraction].copy()
    log(f"rows after missing filter={len(df):,}")

    all_missing = [col for col in df.columns if df[col].isna().all()]
    if all_missing:
        df = df.drop(columns=all_missing)

    if "row_id" in df.columns:
        dedup_subset = [c for c in df.columns if c != "row_id"]
    else:
        dedup_subset = None
    df = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    log(
        f"rows after dedup={len(df):,} | "
        f"benign={(df['binary_label'] == 0).sum():,} | "
        f"attack={(df['binary_label'] == 1).sum():,}"
    )
    return df


def _sort_for_sequences(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.assign(_ts_sort=ts).sort_values(["_ts_sort", "row_id"], na_position="last").drop(columns=["_ts_sort"])
    return df.sort_values("row_id")


def split_detection_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size_from_train: float = 0.20,
    random_state: int = 42,
) -> DetectionSplits:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["binary_label"],
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size_from_train,
        stratify=train_df["binary_label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    log(
        f"split sizes | train={len(train_df):,}, "
        f"val={len(val_df):,}, test={len(test_df):,}"
    )

    train_benign = train_df.loc[train_df["binary_label"] == 0].copy()
    val_benign = val_df.loc[val_df["binary_label"] == 0].copy()

    train_benign = _sort_for_sequences(train_benign).reset_index(drop=True)
    val_benign = _sort_for_sequences(val_benign).reset_index(drop=True)

    return DetectionSplits(
        train_all=train_df,
        val_all=val_df,
        test_all=test_df,
        train_benign=train_benign,
        val_benign=val_benign,
    )


def load_and_prepare_detection_data(
    data_dir: str | Path,
    max_missing_fraction: float = 0.30,
    test_size: float = 0.20,
    val_size_from_train: float = 0.20,
    random_state: int = 42,
    drop_unknown_labels: bool = True,
) -> tuple[pd.DataFrame, DetectionSplits]:
    raw = read_cic_ids2017_folder(data_dir)
    cleaned = clean_detection_dataframe(
        raw,
        max_missing_fraction=max_missing_fraction,
        drop_unknown_labels=drop_unknown_labels,
    )
    splits = split_detection_data(
        cleaned,
        test_size=test_size,
        val_size_from_train=val_size_from_train,
        random_state=random_state,
    )
    return cleaned, splits
