"""Split-conformal anomaly wrapper.

Given a score function s(x) where higher means more anomalous, and a held-out
benign calibration set {z_1, ..., z_n}, this wrapper produces an exchangeable
p-value

    p(x) = (1 + #{i : s(z_i) >= s(x)}) / (n + 1)

Flagging when p(x) <= alpha gives a false-alarm rate bounded by alpha under
the standard conformal exchangeability assumption on benign traffic. The
wrapper is model-agnostic: it works on top of any detector that exposes a
per-row anomaly score (the current RF, the PU detector, an Isolation Forest,
etc.).

Usage pattern:
    wrapper = ConformalAnomalyWrapper(alpha=0.05)
    wrapper.fit(rf_scores_on_val_benign)
    p_test = wrapper.pvalue(rf_scores_on_test)
    preds  = wrapper.predict(rf_scores_on_test)         # 1 when p <= alpha
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


def log(msg: str) -> None:
    print(f"[conformal] {msg}", flush=True)


@dataclass
class ConformalConfig:
    alpha: float = 0.05          # flag when p-value <= alpha
    smoothing: bool = True        # randomized tie-break for exact FAR = alpha
    seed: int = 42


class ConformalAnomalyWrapper:
    def __init__(self, config: Optional[ConformalConfig] = None, alpha: Optional[float] = None):
        cfg = config or ConformalConfig()
        if alpha is not None:
            cfg = ConformalConfig(alpha=alpha, smoothing=cfg.smoothing, seed=cfg.seed)
        self.config = cfg
        self.cal_scores: Optional[np.ndarray] = None  # sorted ascending
        self.derived_threshold: Optional[float] = None

    # ---- calibration ------------------------------------------------------
    def fit(self, cal_scores: np.ndarray) -> "ConformalAnomalyWrapper":
        arr = np.asarray(cal_scores, dtype=float).ravel()
        if arr.size < 20:
            raise ValueError(
                f"Conformal calibration set is too small (n={arr.size}). "
                "Need at least ~20 benign calibration rows."
            )
        self.cal_scores = np.sort(arr)
        self.derived_threshold = float(
            np.quantile(self.cal_scores, 1.0 - self.config.alpha)
        )
        log(
            f"calibrated | n_cal={arr.size:,} | alpha={self.config.alpha} | "
            f"equiv-quantile-threshold={self.derived_threshold:.6f}"
        )
        return self

    # ---- p-values and predictions ----------------------------------------
    def pvalue(self, scores: np.ndarray) -> np.ndarray:
        assert self.cal_scores is not None, "Not calibrated. Call fit(cal_scores) first."
        scores = np.asarray(scores, dtype=float).ravel()
        n = len(self.cal_scores)
        idx = np.searchsorted(self.cal_scores, scores, side="left")
        n_ge = n - idx  # number of calibration scores >= test score

        if not self.config.smoothing:
            return (1.0 + n_ge) / (n + 1.0)

        # Randomized / smoothed conformal p-value to achieve exact FAR = alpha.
        rng = np.random.default_rng(self.config.seed)
        idx_eq = np.searchsorted(self.cal_scores, scores, side="right")
        n_eq = idx_eq - idx
        u = rng.uniform(size=scores.shape)
        return (n_ge - u * n_eq + 1.0) / (n + 1.0)

    def predict(self, scores: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        a = self.config.alpha if alpha is None else float(alpha)
        return (self.pvalue(scores) <= a).astype(int)

    def quantile_threshold(self, alpha: Optional[float] = None) -> float:
        assert self.cal_scores is not None
        a = self.config.alpha if alpha is None else float(alpha)
        return float(np.quantile(self.cal_scores, 1.0 - a))

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "alpha": self.config.alpha,
                "smoothing": self.config.smoothing,
                "seed": self.config.seed,
                "cal_scores": self.cal_scores,
                "derived_threshold": self.derived_threshold,
            },
            path,
        )
        log(f"saved conformal wrapper: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ConformalAnomalyWrapper":
        payload = joblib.load(path)
        cfg = ConformalConfig(
            alpha=payload["alpha"],
            smoothing=payload.get("smoothing", True),
            seed=payload.get("seed", 42),
        )
        obj = cls(cfg)
        obj.cal_scores = payload["cal_scores"]
        obj.derived_threshold = payload.get("derived_threshold")
        return obj
