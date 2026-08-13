#!/usr/bin/env python
"""Test-only evaluation for the trained LSTM log-residual post-processor."""

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LSTM_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = LSTM_ROOT / "train" / "train_lstm.py"

OUTPUT_QCORR = "test_lstm_qcorr_2009_2010.npz"
OUTPUT_METRICS = "test_lstm_metrics_by_station.csv"
OUTPUT_METRICS_SUMMARY = "test_lstm_metrics_summary.csv"
OUTPUT_SKILL = "test_lstm_vs_raw_skill_summary.csv"
OUTPUT_DELTA = "test_only_station_delta.csv"
OUTPUT_SUMMARY = "test_only_evaluation_summary.txt"

REQUIRED_TRAIN_OBJECTS = [
    "LSTMLogResidualCorrector",
    "StationWindowDataset",
    "build_sample_indices",
    "make_dataloader",
    "predict_to_matrix",
    "compute_period_metrics",
    "summarize_metrics",
    "compute_skill_summary",
    "metric_median",
    "load_npz_dataset",
    "validate_arrays",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent 2009–2010 test-only evaluation for a trained LSTM log-residual model."
    )
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pred-clip-abs", type=float, default=None)
    parser.add_argument("--period-name", default="test")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_train_module() -> Any:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")
    spec = importlib.util.spec_from_file_location("lstm_train_module", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training script: {TRAIN_SCRIPT}")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    missing = [name for name in REQUIRED_TRAIN_OBJECTS if not hasattr(train_module, name)]
    if missing:
        raise AttributeError(f"Training script is missing required objects: {', '.join(missing)}")
    return train_module


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        requested = "cpu"
    return torch.device(requested)


def scalar_from_any(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def get_required(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Checkpoint is missing required key: {key}")
    return mapping[key]


def get_config_value(saved_args: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = saved_args.get(key, None)
    return fallback if value is None else value


def resolve_pred_clip_abs(cli_value: Any, saved: Dict[str, Any]) -> float:
    if cli_value is not None:
        return float(cli_value)
    saved_args = saved.get("args", {})
    if isinstance(saved_args, dict) and saved_args.get("pred_clip_abs", None) is not None:
        return float(saved_args["pred_clip_abs"])
    if saved.get("target_clip_abs", None) is not None:
        return float(scalar_from_any(saved["target_clip_abs"]))
    return 8.0


def align_stations(
    npz_station_ids: np.ndarray,
    saved_station_ids: np.ndarray,
    qsim: np.ndarray,
    qobs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz_station_ids = np.asarray(npz_station_ids).astype(np.int64)
    saved_station_ids = np.asarray(saved_station_ids).astype(np.int64)
    if np.array_equal(npz_station_ids, saved_station_ids):
        return saved_station_ids, qsim, qobs

    positions = {int(station_id): idx for idx, station_id in enumerate(npz_station_ids)}
    reorder_indices: List[int] = []
    missing: List[int] = []
    for station_id in saved_station_ids:
        station_id_int = int(station_id)
        if station_id_int not in positions:
            missing.append(station_id_int)
        else:
            reorder_indices.append(positions[station_id_int])
    if missing:
        raise ValueError(f"Checkpoint station_ids not found in NPZ station_ids: {missing[:10]}")

    return saved_station_ids, qsim[:, reorder_indices], qobs[:, reorder_indices]


def compute_test_skill_summary(metrics_df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    raw = metrics_df[(metrics_df["period"] == period_name) & (metrics_df["model"] == "raw")]
    lstm = metrics_df[(metrics_df["period"] == period_name) & (metrics_df["model"] == "lstm")]
    merged = raw.merge(lstm, on=["station_id", "period"], suffixes=("_raw", "_lstm"))

    row: Dict[str, Any] = {"period": period_name}
    for metric in ["corr", "rmse", "nse", "kge"]:
        raw_values = pd.to_numeric(raw[metric], errors="coerce").to_numpy(dtype=np.float64)
        lstm_values = pd.to_numeric(lstm[metric], errors="coerce").to_numpy(dtype=np.float64)
        raw_values = raw_values[np.isfinite(raw_values)]
        lstm_values = lstm_values[np.isfinite(lstm_values)]
        raw_median = np.nan if raw_values.size == 0 else float(np.median(raw_values))
        lstm_median = np.nan if lstm_values.size == 0 else float(np.median(lstm_values))
        row[f"median_raw_{metric}"] = raw_median
        row[f"median_lstm_{metric}"] = lstm_median
        row[f"median_delta_lstm_minus_raw_{metric}"] = (
            np.nan if not (np.isfinite(raw_median) and np.isfinite(lstm_median)) else lstm_median - raw_median
        )

    station_count = int(merged.shape[0])
    if station_count:
        row["stations_improved_corr"] = int((merged["corr_lstm"] > merged["corr_raw"]).sum())
        row["stations_improved_rmse"] = int((merged["rmse_lstm"] < merged["rmse_raw"]).sum())
        row["stations_improved_nse"] = int((merged["nse_lstm"] > merged["nse_raw"]).sum())
        row["stations_improved_kge"] = int((merged["kge_lstm"] > merged["kge_raw"]).sum())
    else:
        row["stations_improved_corr"] = 0
        row["stations_improved_rmse"] = 0
        row["stations_improved_nse"] = 0
        row["stations_improved_kge"] = 0

    for metric in ["corr", "rmse", "nse", "kge"]:
        improved = int(row[f"stations_improved_{metric}"])
        row[f"improved_ratio_{metric}"] = np.nan if station_count == 0 else float(improved / station_count)
    row["station_count"] = station_count
    ordered_columns = [
        "period",
        "median_raw_corr",
        "median_lstm_corr",
        "median_delta_lstm_minus_raw_corr",
        "median_raw_rmse",
        "median_lstm_rmse",
        "median_delta_lstm_minus_raw_rmse",
        "median_raw_nse",
        "median_lstm_nse",
        "median_delta_lstm_minus_raw_nse",
        "median_raw_kge",
        "median_lstm_kge",
        "median_delta_lstm_minus_raw_kge",
        "stations_improved_corr",
        "stations_improved_rmse",
        "stations_improved_nse",
        "stations_improved_kge",
        "improved_ratio_corr",
        "improved_ratio_rmse",
        "improved_ratio_nse",
        "improved_ratio_kge",
        "station_count",
    ]
    return pd.DataFrame([row], columns=ordered_columns)


def compute_station_delta(metrics_df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    raw = metrics_df[(metrics_df["period"] == period_name) & (metrics_df["model"] == "raw")]
    lstm = metrics_df[(metrics_df["period"] == period_name) & (metrics_df["model"] == "lstm")]
    merged = raw.merge(lstm, on=["station_id", "period"], suffixes=("_raw", "_lstm"))
    rows = []
    for _, row in merged.iterrows():
        delta_corr = row["corr_lstm"] - row["corr_raw"]
        delta_rmse = row["rmse_lstm"] - row["rmse_raw"]
        delta_nse = row["nse_lstm"] - row["nse_raw"]
        delta_kge = row["kge_lstm"] - row["kge_raw"]
        rows.append(
            {
                "station_id": int(row["station_id"]),
                "corr_raw": row["corr_raw"],
                "corr_lstm": row["corr_lstm"],
                "rmse_raw": row["rmse_raw"],
                "rmse_lstm": row["rmse_lstm"],
                "nse_raw": row["nse_raw"],
                "nse_lstm": row["nse_lstm"],
                "kge_raw": row["kge_raw"],
                "kge_lstm": row["kge_lstm"],
                "delta_corr": delta_corr,
                "delta_rmse": delta_rmse,
                "delta_nse": delta_nse,
                "delta_kge": delta_kge,
                "improved_corr": bool(row["corr_lstm"] > row["corr_raw"]),
                "improved_rmse": bool(row["rmse_lstm"] < row["rmse_raw"]),
                "improved_nse": bool(row["nse_lstm"] > row["nse_raw"]),
                "improved_kge": bool(row["kge_lstm"] > row["kge_raw"]),
            }
        )
    columns = [
        "station_id",
        "corr_raw",
        "corr_lstm",
        "rmse_raw",
        "rmse_lstm",
        "nse_raw",
        "nse_lstm",
        "kge_raw",
        "kge_lstm",
        "delta_corr",
        "delta_rmse",
        "delta_nse",
        "delta_kge",
        "improved_corr",
        "improved_rmse",
        "improved_nse",
        "improved_kge",
    ]
    return pd.DataFrame(rows, columns=columns)


def finite_date_range(dates: pd.DatetimeIndex, mask: np.ndarray) -> str:
    selected = dates[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        return "none"
    return f"{selected.min().strftime('%Y-%m-%d')} to {selected.max().strftime('%Y-%m-%d')}"


def forward_sanity_check(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> None:
    for batch in loader:
        x, station_idx, _, _, _, _, _ = batch
        x = x.to(device, non_blocking=True)
        station_idx = station_idx.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(x, station_idx)
        if pred.ndim != 1 or pred.shape[0] != x.shape[0]:
            raise ValueError(f"Forward sanity check failed: pred shape {tuple(pred.shape)}, batch size {x.shape[0]}")
        if not torch.isfinite(pred).any():
            raise ValueError("Forward sanity check failed: no finite predictions in first batch")
        return
    raise ValueError("Forward sanity check failed: test loader is empty")


def confirm_outputs_exist(paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected output files were not written: {missing}")


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    dates: pd.DatetimeIndex,
    test_mask_dates: np.ndarray,
    station_count: int,
    test_sample_count: int,
    lookback: int,
    target_clip_abs: float,
    pred_clip_abs: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    station_embedding_dim: int,
    saved_epoch: Any,
    saved_best_val_loss: Any,
    metrics_df: pd.DataFrame,
    skill_df: pd.DataFrame,
    metric_median_func: Any,
    output_paths: List[Path],
) -> None:
    test_raw_corr = metric_median_func(metrics_df, args.period_name, "raw", "corr")
    test_lstm_corr = metric_median_func(metrics_df, args.period_name, "lstm", "corr")
    test_raw_rmse = metric_median_func(metrics_df, args.period_name, "raw", "rmse")
    test_lstm_rmse = metric_median_func(metrics_df, args.period_name, "lstm", "rmse")
    test_raw_nse = metric_median_func(metrics_df, args.period_name, "raw", "nse")
    test_lstm_nse = metric_median_func(metrics_df, args.period_name, "lstm", "nse")
    test_raw_kge = metric_median_func(metrics_df, args.period_name, "raw", "kge")
    test_lstm_kge = metric_median_func(metrics_df, args.period_name, "lstm", "kge")
    skill_row = skill_df.iloc[0].to_dict()

    lines = [
        "This script performs test-only evaluation for 2009–2010.",
        "It loads a trained LSTM checkpoint and does not train or update the model.",
        "The test period is not used for training, validation, scaler fitting, or early stopping.",
        "",
        f"npz_path: {args.npz_path}",
        f"model_path: {args.model_path}",
        f"out_dir: {args.out_dir}",
        f"station_count: {station_count}",
        f"date range: {finite_date_range(dates, test_mask_dates)}",
        f"test_sample_count: {test_sample_count}",
        f"lookback: {lookback}",
        f"target_clip_abs: {target_clip_abs}",
        f"pred_clip_abs: {pred_clip_abs}",
        f"hidden_size: {hidden_size}",
        f"num_layers: {num_layers}",
        f"dropout: {dropout}",
        f"station_embedding_dim: {station_embedding_dim}",
        f"checkpoint_epoch: {saved_epoch}",
        f"checkpoint_best_val_loss: {saved_best_val_loss}",
        "",
        f"test raw median corr/rmse/nse/kge: {test_raw_corr} / {test_raw_rmse} / {test_raw_nse} / {test_raw_kge}",
        f"test LSTM median corr/rmse/nse/kge: {test_lstm_corr} / {test_lstm_rmse} / {test_lstm_nse} / {test_lstm_kge}",
        f"stations_improved_corr and ratio: {skill_row['stations_improved_corr']} / {skill_row['improved_ratio_corr']}",
        f"stations_improved_rmse and ratio: {skill_row['stations_improved_rmse']} / {skill_row['improved_ratio_rmse']}",
        f"stations_improved_nse and ratio: {skill_row['stations_improved_nse']} / {skill_row['improved_ratio_nse']}",
        f"stations_improved_kge and ratio: {skill_row['stations_improved_kge']} / {skill_row['improved_ratio_kge']}",
        "",
        "output files:",
    ]
    lines.extend(str(path) for path in output_paths)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    train_module = load_train_module()
    set_random_seed(args.seed)

    npz_path = Path(args.npz_path)
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {npz_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[DEVICE] requested={args.device}, using={device}", flush=True)
    print(f"[DEVICE] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DEVICE] gpu={torch.cuda.get_device_name(0)}", flush=True)

    saved = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" not in saved:
        raise KeyError("Checkpoint is missing required key: model_state_dict")
    saved_args = saved.get("args", {})
    if not isinstance(saved_args, dict):
        raise TypeError("Checkpoint args must be a dictionary")

    saved_station_ids = np.asarray(get_required(saved, "station_ids")).astype(np.int64)
    x_mean = np.asarray(get_required(saved, "x_mean_by_station"), dtype=np.float32)
    x_std = np.asarray(get_required(saved, "x_std_by_station"), dtype=np.float32)
    if not (len(saved_station_ids) == len(x_mean) == len(x_std)):
        raise ValueError(
            "Checkpoint station_ids, x_mean_by_station, and x_std_by_station lengths must match: "
            f"{len(saved_station_ids)}, {len(x_mean)}, {len(x_std)}"
        )

    lookback = int(scalar_from_any(saved.get("lookback", None), get_config_value(saved_args, "lookback")))
    target_clip_abs = float(scalar_from_any(get_required(saved, "target_clip_abs")))
    pred_clip_abs = resolve_pred_clip_abs(args.pred_clip_abs, saved)
    hidden_size = int(get_config_value(saved_args, "hidden_size"))
    num_layers = int(get_config_value(saved_args, "num_layers"))
    dropout = float(get_config_value(saved_args, "dropout"))
    station_embedding_dim = int(get_config_value(saved_args, "station_embedding_dim"))
    saved_epoch = scalar_from_any(saved.get("epoch", np.nan))
    saved_best_val_loss = scalar_from_any(saved.get("best_val_loss", np.nan))

    data = train_module.load_npz_dataset(npz_path)
    dates = pd.DatetimeIndex(pd.to_datetime(data["dates"].astype(str)))
    npz_station_ids = data["station_ids"].astype(np.int64)
    qsim = data["qsim"].astype(np.float32)
    qobs = data["qobs"].astype(np.float32)
    train_mask_dates = data["train_mask_dates"].astype(bool)
    test_mask_dates = data["test_mask_dates"].astype(bool)
    train_module.validate_arrays(qsim, qobs, train_mask_dates, test_mask_dates)
    if test_mask_dates.shape[0] != qsim.shape[0]:
        raise ValueError("test_mask_dates length must match qsim time dimension")
    qsim = np.where(np.isfinite(qsim), np.maximum(qsim, 0.0), qsim).astype(np.float32)

    station_ids, qsim, qobs = align_stations(npz_station_ids, saved_station_ids, qsim, qobs)
    if qsim.shape[1] != len(saved_station_ids):
        raise ValueError(f"Station count mismatch after alignment: qsim has {qsim.shape[1]}, checkpoint has {len(saved_station_ids)}")

    test_indices = train_module.build_sample_indices(qsim, qobs, test_mask_dates, lookback, target_clip_abs)
    if len(test_indices) <= 0:
        raise ValueError("No test samples were constructed.")
    test_weights = np.ones(len(test_indices), dtype=np.float32)

    test_dataset = train_module.StationWindowDataset(
        qsim=qsim,
        qobs=qobs,
        indices=test_indices,
        sample_weights=test_weights,
        x_mean_by_station=x_mean,
        x_std_by_station=x_std,
        lookback=lookback,
        target_clip_abs=target_clip_abs,
    )
    test_loader = train_module.make_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = train_module.LSTMLogResidualCorrector(
        num_stations=len(saved_station_ids),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        station_embedding_dim=station_embedding_dim,
    )
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()
    forward_sanity_check(model, test_loader, device)

    qcorr = train_module.predict_to_matrix(
        model=model,
        loaders={args.period_name: test_loader},
        qsim=qsim,
        qobs=qobs,
        device=device,
        pred_clip_abs=pred_clip_abs,
    )

    period_masks = {args.period_name: test_mask_dates}
    metrics_df = train_module.compute_period_metrics(qsim, qobs, qcorr, station_ids, period_masks)
    metrics_summary_df = train_module.summarize_metrics(metrics_df)
    skill_df = compute_test_skill_summary(metrics_df, args.period_name)
    delta_df = compute_station_delta(metrics_df, args.period_name)

    qcorr_path = out_dir / OUTPUT_QCORR
    metrics_path = out_dir / OUTPUT_METRICS
    metrics_summary_path = out_dir / OUTPUT_METRICS_SUMMARY
    skill_path = out_dir / OUTPUT_SKILL
    delta_path = out_dir / OUTPUT_DELTA
    summary_path = out_dir / OUTPUT_SUMMARY
    output_paths = [qcorr_path, metrics_path, metrics_summary_path, skill_path, delta_path, summary_path]

    np.savez_compressed(
        qcorr_path,
        dates=np.array([date.strftime("%Y-%m-%d") for date in dates], dtype="U10"),
        station_ids=station_ids,
        qsim=qsim.astype(np.float32),
        qobs=qobs.astype(np.float32),
        qcorr=qcorr.astype(np.float32),
        test_mask_dates=test_mask_dates.astype(bool),
        lookback=np.array(lookback, dtype=np.int64),
        target_mode=np.array("log_residual"),
        target_clip_abs=np.array(target_clip_abs, dtype=np.float32),
        pred_clip_abs=np.array(pred_clip_abs, dtype=np.float32),
        source_model_path=np.array(str(model_path)),
        checkpoint_epoch=np.array(saved_epoch),
        checkpoint_best_val_loss=np.array(saved_best_val_loss),
    )
    metrics_df.to_csv(metrics_path, index=False)
    metrics_summary_df.to_csv(metrics_summary_path, index=False)
    skill_df.to_csv(skill_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    write_summary(
        summary_path=summary_path,
        args=args,
        dates=dates,
        test_mask_dates=test_mask_dates,
        station_count=len(station_ids),
        test_sample_count=len(test_indices),
        lookback=lookback,
        target_clip_abs=target_clip_abs,
        pred_clip_abs=pred_clip_abs,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        station_embedding_dim=station_embedding_dim,
        saved_epoch=saved_epoch,
        saved_best_val_loss=saved_best_val_loss,
        metrics_df=metrics_df,
        skill_df=skill_df,
        metric_median_func=train_module.metric_median,
        output_paths=output_paths,
    )
    confirm_outputs_exist(output_paths)

    print("[DONE] test-only evaluation written:", flush=True)
    for name in [OUTPUT_QCORR, OUTPUT_METRICS, OUTPUT_METRICS_SUMMARY, OUTPUT_SKILL, OUTPUT_DELTA, OUTPUT_SUMMARY]:
        print(f"- {name}", flush=True)
    print("This script performs test-only evaluation for 2009–2010. It does not train or update the model.", flush=True)


if __name__ == "__main__":
    main()
