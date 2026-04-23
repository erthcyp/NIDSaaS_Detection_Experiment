from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


HTTP_PORTS = {80, 443, 8080, 8000, 8443}
BRUTE_FORCE_PORTS = {21, 22, 23, 80, 443, 3389}


@dataclass
class DataConfig:
    data_dir: str
    output_dir: str = "outputs_detection"
    random_state: int = 42
    test_size: float = 0.20
    val_size_from_train: float = 0.20
    max_missing_fraction: float = 0.30
    drop_unknown_labels: bool = True
    save_split_indices: bool = True

@dataclass
class RFConfig:
    n_svd_components: int = 64
    n_rff_components: int = 192
    rff_gamma: float = 0.10
    n_rotations: int = 8

    n_estimators: int = 200
    max_depth: Optional[int] = 20
    min_samples_leaf: int = 2
    n_jobs: int = 8

    threshold_quantile: float = 0.97
    calibrated_threshold: Optional[float] = 0.8042
    use_calibrated_threshold: bool = False

    exclude_columns: Tuple[str, ...] = (
        "Flow ID",
        "flow_id",
        "Source IP",
        "source_ip",
        "Destination IP",
        "destination_ip",
        "Timestamp",
        "timestamp",
        "SimillarHTTP",
        "simillarhttp",
    )


@dataclass
class LSTMConfig:
    seq_len: int = 10
    hidden_size: int = 64
    latent_dim: int = 32
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 15
    lr: float = 1e-3
    threshold_quantile: float = 0.95
    calibrated_threshold: Optional[float] = 0.8226
    use_calibrated_threshold: bool = True
    svd_components: int = 64
    sort_by_time: bool = True
    device: str = "cpu"
    exclude_columns: Tuple[str, ...] = (
        "Flow ID",
        "flow_id",
        "Source IP",
        "source_ip",
        "Destination IP",
        "destination_ip",
        "Timestamp",
        "timestamp",
        "SimillarHTTP",
        "simillarhttp",
    )


@dataclass
class ExperimentConfig:
    data: DataConfig
    rf: RFConfig = field(default_factory=RFConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
