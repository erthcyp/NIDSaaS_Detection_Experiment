from __future__ import annotations

from dataclasses import asdict

import gc
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler

from config import RFConfig
from features import build_tabular_preprocessor, select_feature_columns
from utils import set_random_seed


def log(msg: str) -> None:
    print(f"[rf_anomaly] {msg}", flush=True)


class SelfSupervisedRFAnomaly:
    """Benign-only anomaly detector aligned to the paper draft.

    Notes for this patched/debug version:
      - Keeps the same general pipeline: preprocessor -> SVD -> RFF -> self-supervised RF
      - Adds downsampling for benign train/val to make local runs tractable
      - Adds memory/timing logs
      - Uses lighter RF settings so the pipeline can complete on a normal machine
    """

    def __init__(self, config: RFConfig | None = None):
        self.config = config or RFConfig()
        self.feature_columns: list[str] | None = None
        self.preprocessor = None
        self.svd = None
        self.rff = None
        self.rf = None
        self.rotations: np.ndarray | None = None
        self.threshold: float | None = None
        self.derived_threshold: float | None = None

    def _to_dense(self, x):
        return x.toarray() if sparse.issparse(x) else np.asarray(x)

    def _make_rotations(self, dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rotations = [np.eye(dim, dtype=np.float32)]
        for _ in range(self.config.n_rotations - 1):
            q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            rotations.append(q.astype(np.float32))
        return np.stack(rotations, axis=0)

    def _apply_transform_chain(self, df: pd.DataFrame) -> np.ndarray:
        x = self.preprocessor.transform(df[self.feature_columns])
        x = self._to_dense(x)
        x = self.svd.transform(x)
        x = self.rff.transform(x)
        return x.astype(np.float32)

    def _build_self_supervised_dataset(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        d = x.shape[1]
        m = len(self.rotations)

        xs = np.empty((n * m, d), dtype=np.float32)
        ys = np.empty(n * m, dtype=np.int64)

        for class_idx, rot in enumerate(self.rotations):
            start = class_idx * n
            end = (class_idx + 1) * n
            xs[start:end] = x @ rot
            ys[start:end] = class_idx

        return xs, ys

    def fit(
        self,
        train_benign_df: pd.DataFrame,
        val_benign_df: pd.DataFrame,
        random_state: int = 42,
    ) -> "SelfSupervisedRFAnomaly":
        set_random_seed(random_state)
        log(
            f"fit start | train_benign={len(train_benign_df):,}, "
            f"val_benign={len(val_benign_df):,}, seed={random_state}"
        )

        # Downsample first so local training finishes in a reasonable time.
        max_train_benign = 450_000
        max_val_benign = 120_000

        if len(train_benign_df) > max_train_benign:
            train_benign_df = (
                train_benign_df
                .sample(n=max_train_benign, random_state=random_state)
                .reset_index(drop=True)
            )
            log(f"downsampled train_benign to {len(train_benign_df):,}")

        if len(val_benign_df) > max_val_benign:
            val_benign_df = (
                val_benign_df
                .sample(n=max_val_benign, random_state=random_state)
                .reset_index(drop=True)
            )
            log(f"downsampled val_benign to {len(val_benign_df):,}")

        self.feature_columns = select_feature_columns(
            train_benign_df,
            exclude_columns=self.config.exclude_columns,
        )
        log(f"selected {len(self.feature_columns)} feature columns")

        self.preprocessor = build_tabular_preprocessor(
                train_benign_df,
                feature_columns=self.feature_columns,
                scale_numeric=True,
            )

        x_train_base = self.preprocessor.fit_transform(train_benign_df[self.feature_columns])
        x_train_base = self._to_dense(x_train_base).astype(np.float32, copy=False)
        log(f"preprocessed train shape={x_train_base.shape}")

        n_svd = min(self.config.n_svd_components, max(2, x_train_base.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_svd, random_state=random_state)
        x_train_svd = self.svd.fit_transform(x_train_base).astype(np.float32, copy=False)
        log(f"after SVD shape={x_train_svd.shape}")

        self.rff = RBFSampler(
            gamma=self.config.rff_gamma,
            n_components=self.config.n_rff_components,
            random_state=random_state,
        )
        x_train_rff = self.rff.fit_transform(x_train_svd).astype(np.float32, copy=False)
        log(f"after RFF shape={x_train_rff.shape}")

        self.rotations = self._make_rotations(x_train_rff.shape[1], seed=random_state)
        x_train_ss, y_train_ss = self._build_self_supervised_dataset(x_train_rff)
        log(
            f"self-supervised dataset shape={x_train_ss.shape}, "
            f"labels={len(np.unique(y_train_ss))}"
        )

        approx_gb = x_train_ss.nbytes / (1024 ** 3)
        log(f"self-supervised matrix memory ~ {approx_gb:.2f} GB")

        # Free intermediate arrays before RF training to reduce memory pressure.
        del x_train_base, x_train_svd, x_train_rff
        gc.collect()

        # Lighter RF settings for a tractable local debug run.
        #rf_n_estimators = 50
        #rf_max_depth = 16
        #rf_min_samples_leaf = 5

        self.rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=self.config.n_jobs,
            random_state=random_state,
        )
        log(
            f"training RandomForest | n_estimators={self.config.n_estimators}, "
            f"max_depth={self.config.max_depth}, min_samples_leaf={self.config.min_samples_leaf}, "
            f"n_jobs={self.config.n_jobs}"
        )

        start_fit = time.time()
        self.rf.fit(x_train_ss, y_train_ss)
        elapsed_fit = time.time() - start_fit
        log(f"RandomForest training finished in {elapsed_fit:.2f}s")

        # Free the large self-supervised matrices after fitting.
        del x_train_ss, y_train_ss
        gc.collect()

        val_scores = self.score_samples(val_benign_df)
        derived_threshold = float(np.quantile(val_scores, self.config.threshold_quantile))
        self.threshold = (
            float(self.config.calibrated_threshold)
            if self.config.use_calibrated_threshold and self.config.calibrated_threshold is not None
            else derived_threshold
        )
        self.derived_threshold = derived_threshold
        log(
            f"validation done | derived_threshold={self.derived_threshold:.6f}, "
            f"using_threshold={self.threshold:.6f}"
        )
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        if self.rf is None:
            raise RuntimeError("Model is not fitted yet.")

        x = self._apply_transform_chain(df)
        log(f"scoring {len(df):,} rows ...")

        normality_sum = np.zeros(x.shape[0], dtype=np.float32)

        for class_idx, rot in enumerate(self.rotations):
            x_rot = x @ rot
            probs = self.rf.predict_proba(x_rot)[:, class_idx]
            normality_sum += probs.astype(np.float32, copy=False)

        normality = normality_sum / len(self.rotations)
        anomaly_scores = 1.0 - normality
        return anomaly_scores.astype(float)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_samples(df)
        preds = (scores > self.threshold).astype(int)
        return preds, scores

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "config": asdict(self.config),
                "feature_columns": self.feature_columns,
                "preprocessor": self.preprocessor,
                "svd": self.svd,
                "rff": self.rff,
                "rotations": self.rotations,
                "rf": self.rf,
                "threshold": self.threshold,
                "derived_threshold": self.derived_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SelfSupervisedRFAnomaly":
        payload = joblib.load(path)
        model = cls(RFConfig(**payload["config"]))
        model.feature_columns = payload["feature_columns"]
        model.preprocessor = payload["preprocessor"]
        model.svd = payload["svd"]
        model.rff = payload["rff"]
        model.rotations = payload["rotations"]
        model.rf = payload["rf"]
        model.threshold = payload["threshold"]
        model.derived_threshold = payload.get("derived_threshold")
        return model
