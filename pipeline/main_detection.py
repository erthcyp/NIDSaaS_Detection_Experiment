from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import DataConfig, ExperimentConfig
from load_data import load_and_prepare_detection_data
from metrics import binary_metrics, class_wise_detection
from rf_anomaly import SelfSupervisedRFAnomaly
from utils import make_dir, set_random_seed, write_json


MODEL_NAME_MAP = {
    "signature": "Signature",
    "rf": "RF",
    "lstm": "LSTM",
    "hybrid": "Hybrid",
}
def log(msg: str) -> None:
    print(f"[main_detection] {msg}", flush=True)


def save_bar_plot(metrics_df: pd.DataFrame, out_path: Path) -> None:
    metric_order = ["accuracy", "precision", "recall", "f1"]
    plot_df = metrics_df.set_index("paper_model")[metric_order]
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title("Detection performance comparison")
    ax.legend(loc="lower right")
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_experiment(config: ExperimentConfig, mode: str = "all") -> None:
    out_dir = Path(config.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"ensured output directory exists: {out_dir.resolve()}")
    set_random_seed(config.data.random_state)
    log(f"output_dir={out_dir}")
    log(f"seed={config.data.random_state}")
    log(f"mode={mode}")
    log("loading and preparing data ...")
    cleaned_df, splits = load_and_prepare_detection_data(
        config.data.data_dir,
        max_missing_fraction=config.data.max_missing_fraction,
        test_size=config.data.test_size,
        val_size_from_train=config.data.val_size_from_train,
        random_state=config.data.random_state,
        drop_unknown_labels=config.data.drop_unknown_labels,
    )
    log(
    "data ready: "
    f"cleaned={len(cleaned_df):,}, "
    f"train={len(splits.train_all):,}, "
    f"val={len(splits.val_all):,}, "
    f"test={len(splits.test_all):,}, "
    f"train_benign={len(splits.train_benign):,}, "
    f"val_benign={len(splits.val_benign):,}"
    )

    results = []
    pred_cache: dict[str, pd.Series] = {}

    y_test = splits.test_all["binary_label"].to_numpy()
    multiclass_test = splits.test_all["multiclass_label"]
    
    
    if mode in {"all", "signature", "hybrid"}:
        from signature import SignaturePrefilter
        log("running signature prefilter ...")
        t0 = time.perf_counter()
        sig = SignaturePrefilter(config.signature)
        y_sig, sig_rule_hits = sig.predict(splits.test_all)
        elapsed = time.perf_counter() - t0
        sig_metrics = binary_metrics(y_test, y_sig)
        sig_metrics.update({"model": "signature", "paper_model": MODEL_NAME_MAP["signature"], "total_time_s": elapsed})
        results.append(sig_metrics)
        pred_cache["signature"] = pd.Series(y_sig)
        sig_rule_hits.to_csv(out_dir / "signature_rule_hits_test.csv", index=False)
        sig_rule_hits.sum().rename("hit_count").to_csv(out_dir / "signature_rule_summary.csv", header=True)
        write_json(sig.get_params(), out_dir / "signature_config.json")
        log(f"signature done in {elapsed:.2f}s")
        
    if mode in {"all", "rf", "hybrid"}:
        log("running RF anomaly detector ...")
        t0 = time.perf_counter()
        rf_model = SelfSupervisedRFAnomaly(config.rf).fit(
            splits.train_benign,
            splits.val_benign,
            random_state=config.data.random_state,
        )
        y_rf, rf_scores = rf_model.predict(splits.test_all)
        rf_pred_df = splits.test_all.copy()
        rf_pred_df["rf_pred"] = y_rf
        rf_pred_df["rf_score"] = rf_scores
        rf_pred_df.to_csv(out_dir / "rf_predictions.csv", index=False)
        log(f"saved RF predictions: {out_dir / 'rf_predictions.csv'}")
        
        elapsed = time.perf_counter() - t0
        rf_metrics = binary_metrics(y_test, y_rf, scores=rf_scores)
        rf_metrics.update(
            {
                "model": "rf",
                "paper_model": MODEL_NAME_MAP["rf"],
                "total_time_s": elapsed,
                "threshold": rf_model.threshold,
                "derived_threshold": rf_model.derived_threshold,
            }
        )
        results.append(rf_metrics)
        pred_cache["rf"] = pd.Series(y_rf)
        rf_model.save(out_dir / "rf_anomaly.joblib")
        log(
            f"rf done in {elapsed:.2f}s | "
            f"threshold={rf_model.threshold:.6f} | "
            f"derived_threshold={rf_model.derived_threshold:.6f}"
        )

    if mode in {"all", "lstm"}:
        from lstm_anomaly import LSTMAnomalyDetector
        log("running LSTM anomaly detector ...")
        t0 = time.perf_counter()
        lstm_model = LSTMAnomalyDetector(config.lstm).fit(
            splits.train_benign,
            splits.val_benign,
            random_state=config.data.random_state,
        )
        y_lstm, lstm_scores = lstm_model.predict(splits.test_all)
        elapsed = time.perf_counter() - t0
        lstm_metrics = binary_metrics(y_test, y_lstm, scores=lstm_scores)
        lstm_metrics.update(
            {
                "model": "lstm",
                "paper_model": MODEL_NAME_MAP["lstm"],
                "total_time_s": elapsed,
                "threshold": lstm_model.threshold,
                "derived_threshold": lstm_model.derived_threshold,
            }
        )
        results.append(lstm_metrics)
        pred_cache["lstm"] = pd.Series(y_lstm)
        lstm_model.save(out_dir / "lstm_anomaly.joblib")
        log(
            f"lstm done in {elapsed:.2f}s | "
            f"threshold={lstm_model.threshold:.6f} | "
            f"derived_threshold={lstm_model.derived_threshold:.6f}"
        )

    if mode in {"all", "hybrid"}:
        log("running hybrid OR merge ...")
        if "signature" not in pred_cache or "rf" not in pred_cache:
            raise RuntimeError("Hybrid mode requires signature and rf predictions.")
        t0 = time.perf_counter()
        y_hybrid = or_hybrid(pred_cache["signature"].to_numpy(), pred_cache["rf"].to_numpy())
        elapsed = time.perf_counter() - t0
        hy_metrics = binary_metrics(y_test, y_hybrid)
        hy_metrics.update({"model": "hybrid", "paper_model": MODEL_NAME_MAP["hybrid"], "total_time_s": elapsed})
        results.append(hy_metrics)
        pred_cache["hybrid"] = pd.Series(y_hybrid)

        classwise = class_wise_detection(multiclass_test, y_hybrid)
        classwise.to_csv(out_dir / "classwise_hybrid.csv", index=False)
        log(f"hybrid done in {elapsed:.2f}s")

    metrics_df = pd.DataFrame(results)
    column_order = [
        "paper_model", "model", "accuracy", "precision", "recall", "f1", "far",
        "roc_auc", "pr_auc", "tp", "tn", "fp", "fn", "threshold", "derived_threshold", "total_time_s",
    ]
    metrics_df = metrics_df[[c for c in column_order if c in metrics_df.columns]]
    log("saving outputs ...")
    metrics_df.to_csv(out_dir / "overall_metrics.csv", index=False)
    save_bar_plot(metrics_df, out_dir / "detection_performance.png")

    split_summary = {
        "n_total": int(len(cleaned_df)),
        "n_train": int(len(splits.train_all)),
        "n_val": int(len(splits.val_all)),
        "n_test": int(len(splits.test_all)),
        "n_train_benign": int(len(splits.train_benign)),
        "n_val_benign": int(len(splits.val_benign)),
        "binary_distribution_train": splits.train_all["binary_label"].value_counts().to_dict(),
        "binary_distribution_val": splits.val_all["binary_label"].value_counts().to_dict(),
        "binary_distribution_test": splits.test_all["binary_label"].value_counts().to_dict(),
    }
    write_json(split_summary, out_dir / "split_summary.json")

    if config.data.save_split_indices and "row_id" in cleaned_df.columns:
        split_indices = {
            "train_row_id": splits.train_all["row_id"].tolist(),
            "val_row_id": splits.val_all["row_id"].tolist(),
            "test_row_id": splits.test_all["row_id"].tolist(),
        }
        write_json(split_indices, out_dir / "split_row_ids.json")

    log(f"saved outputs to: {out_dir}")
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection experiments for the NIDSaaS paper draft.")
    parser.add_argument("--data-dir", required=True, help="Directory containing CIC-IDS2017 CSV files.")
    parser.add_argument("--output-dir", default="outputs_detection", help="Directory for results and models.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "signature", "rf", "lstm", "hybrid"],
        help="Which experiment to run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data=DataConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            random_state=args.seed,
        )
    )
    run_experiment(cfg, mode=args.mode)
