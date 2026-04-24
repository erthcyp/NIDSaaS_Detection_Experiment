#!/usr/bin/env python3
"""
lstm_autoencoder_baseline.py
============================

Lightweight LSTM-autoencoder anomaly detector for the baseline-comparison
script. Self-contained PyTorch implementation; no dependency on any removed
project module.

Model
-----
Encoder:   LSTM(input_dim -> hidden) -> last-hidden linear -> latent
Decoder:   latent tiled seq_len times -> LSTM(latent -> hidden)
           -> linear(hidden -> input_dim) applied per timestep

Training
--------
Benign-only. MSELoss on full-sequence reconstruction. Adam optimiser.

Scoring
-------
Per-sequence reconstruction MSE on the LAST timestep (aligns the score with
the target flow identity), yielding one anomaly score per flow ending a
valid sliding window. The first (seq_len - 1) flows reuse the first valid
score, so output shape matches input row count.

Usage
-----
    scores = lstm_autoencoder_scores(
        X_train_benign=X_train_benign,
        X_test=X_test,
        seq_len=10,
        hidden_size=64,
        latent_dim=32,
        epochs=8,
        batch_size=256,
        lr=1e-3,
        train_size=200_000,
        device="cpu",
        seed=42,
    )
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np


def log(msg: str) -> None:
    print(f"[lstm-ae] {msg}", flush=True)


# -------------------------------------------------------------------------
# sliding-window sequence helpers
# -------------------------------------------------------------------------

def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Sliding windows of length `seq_len` over rows of X.

    Returns array of shape (n_windows, seq_len, n_features) where
    n_windows = X.shape[0] - seq_len + 1.
    """
    n, d = X.shape
    if n < seq_len:
        raise ValueError(
            f"Need at least seq_len={seq_len} rows to build sequences, got {n}."
        )
    n_windows = n - seq_len + 1
    # As-strided view = O(1) memory but unsafe if downstream code writes.
    # Use fancy indexing for clarity and predictability; cost is acceptable
    # given sizes we operate on.
    out = np.empty((n_windows, seq_len, d), dtype=X.dtype)
    for t in range(seq_len):
        out[:, t, :] = X[t : t + n_windows]
    return out


def align_scores_to_rows(
    seq_scores: np.ndarray, n_rows: int, seq_len: int
) -> np.ndarray:
    """Expand per-sequence scores back to per-row scores.

    Sequence i covers rows [i, i + seq_len - 1]; we attribute its score to
    row (i + seq_len - 1). Rows 0 .. seq_len-2 (no full window) copy the
    first valid score.
    """
    if seq_scores.shape[0] == 0:
        return np.zeros(n_rows, dtype=np.float64)
    row_scores = np.empty(n_rows, dtype=np.float64)
    head = seq_len - 1
    row_scores[:head] = seq_scores[0]
    row_scores[head : head + seq_scores.shape[0]] = seq_scores
    return row_scores


# -------------------------------------------------------------------------
# LSTM-autoencoder
# -------------------------------------------------------------------------

def _build_model(input_dim: int, hidden_size: int, latent_dim: int, seq_len: int):
    import torch
    from torch import nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq_len = seq_len
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.encoder_to_latent = nn.Linear(hidden_size, latent_dim)
            self.decoder = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, input_dim)

        def forward(self, x):  # x: (B, T, D)
            _, (h, _) = self.encoder(x)          # h: (1, B, H)
            z = self.encoder_to_latent(h[-1])     # (B, latent)
            z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, latent)
            y, _ = self.decoder(z_seq)            # (B, T, H)
            return self.output(y)                 # (B, T, D)

    return LSTMAutoencoder()


def lstm_autoencoder_scores(
    X_train_benign: np.ndarray,
    X_test: np.ndarray,
    seq_len: int = 10,
    hidden_size: int = 64,
    latent_dim: int = 32,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    train_size: Optional[int] = 200_000,
    device: str = "cpu",
    seed: int = 42,
    X_val: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], np.ndarray, float, float]:
    """Fit an LSTM autoencoder on benign training sequences and return
    per-row anomaly scores over (optionally) X_val and X_test.

    If ``X_val`` is provided, it must already be ordered by time (the same
    way X_test is) so that sliding windows are temporally coherent.

    Returns
    -------
    val_scores    : (X_val.shape[0],) float64 array -- or ``None`` if
                    ``X_val`` was not provided.
    test_scores   : (X_test.shape[0],) float64 array
    fit_seconds   : float
    score_seconds : float (test-set scoring time only, for like-for-like
                    latency comparison with iForest / OC-SVM)
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "lstm_autoencoder_baseline requires PyTorch. "
            "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from e

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---- training data ----
    X_tr = X_train_benign.astype(np.float32, copy=False)
    if train_size is not None and X_tr.shape[0] > train_size:
        # Preserve temporal contiguity: take a *consecutive* slice rather
        # than a random subsample, otherwise sliding windows straddle
        # arbitrary time jumps and the LSTM learns noise.
        start = rng.integers(0, X_tr.shape[0] - train_size + 1)
        X_tr = X_tr[start : start + train_size]
        log(
            f"subsampled benign training rows to {X_tr.shape[0]:,} "
            f"(consecutive slice, preserves temporal locality)"
        )

    log(
        f"building training sequences | rows={X_tr.shape[0]:,} "
        f"seq_len={seq_len} features={X_tr.shape[1]}"
    )
    X_tr_seq = make_sequences(X_tr, seq_len)
    log(f"training sequences: {X_tr_seq.shape}")

    input_dim = X_tr_seq.shape[2]
    dev = torch.device(device)
    model = _build_model(input_dim, hidden_size, latent_dim, seq_len).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tensor_train = torch.from_numpy(X_tr_seq)
    loader = DataLoader(
        TensorDataset(tensor_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    log(
        f"training LSTM-AE | hidden={hidden_size} latent={latent_dim} "
        f"epochs={epochs} batch={batch_size} lr={lr} device={device}"
    )
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        running = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(dev, non_blocking=True)
            optimiser.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimiser.step()
            running += float(loss.item())
            n_batches += 1
        avg = running / max(n_batches, 1)
        log(f"  epoch {ep + 1:2d}/{epochs} | mean MSE = {avg:.6f}")
    fit_s = time.perf_counter() - t0

    # ---- scoring helper (reused for val + test) ----
    model.eval()
    eval_batch = 1024

    def _score_matrix(X: np.ndarray, tag: str) -> tuple[np.ndarray, float]:
        log(f"scoring {tag} | rows={X.shape[0]:,} seq_len={seq_len}")
        X_f = X.astype(np.float32, copy=False)
        X_seq = make_sequences(X_f, seq_len)
        log(f"{tag} sequences: {X_seq.shape}")
        n_windows = X_seq.shape[0]
        seq_scores = np.empty(n_windows, dtype=np.float64)
        t_start = time.perf_counter()
        with torch.no_grad():
            for start in range(0, n_windows, eval_batch):
                end = min(start + eval_batch, n_windows)
                b = torch.from_numpy(X_seq[start:end]).to(
                    dev, non_blocking=True
                )
                recon = model(b)
                # Reconstruction MSE on the LAST timestep only -> per-flow.
                last_true = b[:, -1, :]
                last_recon = recon[:, -1, :]
                mse = ((last_true - last_recon) ** 2).mean(dim=1).cpu().numpy()
                seq_scores[start:end] = mse
        elapsed = time.perf_counter() - t_start
        aligned = align_scores_to_rows(seq_scores, X.shape[0], seq_len)
        return aligned, elapsed

    # ---- score val (optional) + test ----
    if X_val is not None:
        val_aligned, _ = _score_matrix(X_val, tag="val set")
    else:
        val_aligned = None

    test_aligned, score_s = _score_matrix(X_test, tag="test set")
    return val_aligned, test_aligned, fit_s, score_s
