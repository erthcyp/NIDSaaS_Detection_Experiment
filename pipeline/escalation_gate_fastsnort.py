"""Escalation gate for the FastSnort cascade.

This variant is intended for the split-calibration cascade where Snort is used
only as a fast-path short-circuit, not as a meta-feature inside the gate.

Gate meta-features (determined at fit time from the meta DataFrame columns):
    - rf_score
    - rf_pvalue
    - optional tier-2 rate-rule indicators (rate_L, rate_P, ...)

The gate records whichever meta columns were present at fit time and
requires the same columns at predict time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier


def log(msg: str) -> None:
    print(f"[escalation_gate_fastsnort] {msg}", flush=True)


@dataclass
class EscalationGateFastSnortConfig:
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: Optional[int] = 8
    l2_regularization: float = 1.0
    min_samples_leaf: int = 40
    early_stopping: bool = True
    validation_fraction: float = 0.15
    class_weight: Optional[str] = "balanced"
    threshold: float = 0.5
    random_state: int = 42


REQUIRED_META_FEATURES = ("rf_score", "rf_pvalue")


def _to_dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _assemble_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    preprocessor,
    meta: pd.DataFrame,
    meta_columns: list[str],
) -> np.ndarray:
    x_feat = preprocessor.transform(df[feature_columns])
    x_feat = _to_dense(x_feat).astype(np.float32, copy=False)
    missing = [c for c in meta_columns if c not in meta.columns]
    if missing:
        raise ValueError(f"meta DataFrame is missing required columns: {missing}")
    meta_arr = meta[list(meta_columns)].to_numpy(dtype=np.float32, copy=False)
    if np.isnan(meta_arr).any():
        meta_arr = np.nan_to_num(meta_arr, nan=0.0)
    return np.hstack([x_feat, meta_arr]).astype(np.float32, copy=False)


class EscalationGateFastSnort:
    def __init__(self, config: Optional[EscalationGateFastSnortConfig] = None):
        self.config = config or EscalationGateFastSnortConfig()
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.feature_columns: Optional[list[str]] = None
        self.meta_columns: Optional[list[str]] = None
        self.preprocessor = None
        self.trained_on_n: Optional[int] = None
        self.pos_rate_train: Optional[float] = None

    def fit(
        self,
        df: pd.DataFrame,
        meta: pd.DataFrame,
        y: np.ndarray,
        feature_columns: list[str],
        preprocessor,
    ) -> "EscalationGateFastSnort":
        if len(df) != len(meta) or len(df) != len(y):
            raise ValueError("df, meta, and y must have the same length.")
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError(
                "Escalation pool has a single class; widen alpha_escalate to "
                "include both benign and attack examples for training."
            )

        self.feature_columns = list(feature_columns)
        self.preprocessor = preprocessor

        # Meta columns: required ones first (rf_score, rf_pvalue), then any
        # additional numeric columns the caller provided -- e.g. tier-2
        # rate-rule indicators rate_L, rate_P. Preserved order so load/save
        # round-trips produce identical feature matrices.
        missing_required = [c for c in REQUIRED_META_FEATURES if c not in meta.columns]
        if missing_required:
            raise ValueError(
                f"meta DataFrame must contain {REQUIRED_META_FEATURES}; missing {missing_required}"
            )
        extra = [c for c in meta.columns if c not in REQUIRED_META_FEATURES]
        self.meta_columns = list(REQUIRED_META_FEATURES) + extra
        log(f"gate meta columns: {self.meta_columns}")

        x = _assemble_features(
            df, self.feature_columns, self.preprocessor, meta, self.meta_columns
        )
        log(
            f"training gate | n={x.shape[0]:,}, d={x.shape[1]:,}, "
            f"pos_rate={y.mean():.4f}"
        )

        sample_weight = None
        if self.config.class_weight == "balanced":
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos > 0 and n_neg > 0:
                w_pos = 0.5 * (n_pos + n_neg) / n_pos
                w_neg = 0.5 * (n_pos + n_neg) / n_neg
                sample_weight = np.where(y == 1, w_pos, w_neg).astype(np.float32)

        self.model = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            l2_regularization=self.config.l2_regularization,
            min_samples_leaf=self.config.min_samples_leaf,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
            random_state=self.config.random_state,
        )
        self.model.fit(x, y, sample_weight=sample_weight)
        self.trained_on_n = int(len(y))
        self.pos_rate_train = float(y.mean())
        log(
            f"gate trained | n_iter={self.model.n_iter_} | "
            f"trained_on_n={self.trained_on_n:,}"
        )
        return self

    def predict_proba(self, df: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Gate not trained."
        assert self.meta_columns is not None, "Gate not trained (no meta_columns)."
        x = _assemble_features(
            df, self.feature_columns, self.preprocessor, meta, self.meta_columns
        )
        return self.model.predict_proba(x)[:, 1].astype(float)

    def predict(
        self,
        df: pd.DataFrame,
        meta: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        t = self.config.threshold if threshold is None else float(threshold)
        probs = self.predict_proba(df, meta)
        preds = (probs >= t).astype(int)
        return preds, probs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": asdict(self.config),
                "feature_columns": self.feature_columns,
                "meta_columns": self.meta_columns,
                "preprocessor": self.preprocessor,
                "model": self.model,
                "trained_on_n": self.trained_on_n,
                "pos_rate_train": self.pos_rate_train,
            },
            path,
        )
        log(f"saved escalation gate: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EscalationGateFastSnort":
        payload = joblib.load(path)
        cfg_fields = set(EscalationGateFastSnortConfig.__dataclass_fields__.keys())
        cfg_dict = {k: v for k, v in payload["config"].items() if k in cfg_fields}
        obj = cls(EscalationGateFastSnortConfig(**cfg_dict))
        obj.feature_columns = payload["feature_columns"]
        obj.meta_columns = payload.get("meta_columns") or list(REQUIRED_META_FEATURES)
        obj.preprocessor = payload["preprocessor"]
        obj.model = payload["model"]
        obj.trained_on_n = payload.get("trained_on_n")
        obj.pos_rate_train = payload.get("pos_rate_train")
        return obj
