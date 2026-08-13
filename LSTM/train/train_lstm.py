#!/usr/bin/env python
"""Train an offline LSTM log-residual post-processor for station Qsim/Qobs."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
LSTM_ROOT = SCRIPT_DIR.parent

REGIME_NAMES = ["low", "lower", "normal", "higher", "high"]
METRIC_NAMES = [
    "corr",
    "rmse",
    "mae",
    "nse",
    "lognse",
    "kge",
    "kgess",
    "bias",
    "pbias",
    "mean_ratio",
]
SUMMARY_METRICS = METRIC_NAMES


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_pair_mask(sim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return np.isfinite(sim) & np.isfinite(obs)


def safe_corr(sim: np.ndarray, obs: np.ndarray, eps: float = 1e-6) -> float:
    mask = finite_pair_mask(sim, obs)
    if int(mask.sum()) < 3:
        return np.nan
    sim_valid = sim[mask]
    obs_valid = obs[mask]
    if np.std(sim_valid) <= eps or np.std(obs_valid) <= eps:
        return np.nan
    return float(np.corrcoef(sim_valid, obs_valid)[0, 1])


def safe_nse(sim: np.ndarray, obs: np.ndarray, eps: float = 1e-6) -> float:
    mask = finite_pair_mask(sim, obs)
    if int(mask.sum()) < 3:
        return np.nan
    sim_valid = sim[mask]
    obs_valid = obs[mask]
    denom = float(np.sum((obs_valid - np.mean(obs_valid)) ** 2))
    if denom <= eps:
        return np.nan
    return float(1.0 - np.sum((sim_valid - obs_valid) ** 2) / denom)


def safe_lognse(sim: np.ndarray, obs: np.ndarray, log_eps: float = 0.1, eps: float = 1e-6) -> float:
    mask = finite_pair_mask(sim, obs) & ((sim + log_eps) > 0.0) & ((obs + log_eps) > 0.0)
    if int(mask.sum()) < 3:
        return np.nan
    return safe_nse(np.log(sim[mask] + log_eps), np.log(obs[mask] + log_eps), eps=eps)


def safe_kge(sim: np.ndarray, obs: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    mask = finite_pair_mask(sim, obs)
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    sim_valid = sim[mask]
    obs_valid = obs[mask]
    corr = safe_corr(sim_valid, obs_valid, eps=eps)
    obs_std = float(np.std(obs_valid))
    obs_mean = float(np.mean(obs_valid))
    if not np.isfinite(corr) or obs_std <= eps or abs(obs_mean) <= eps:
        return np.nan, np.nan
    alpha = float(np.std(sim_valid) / obs_std)
    beta = float(np.mean(sim_valid) / obs_mean)
    kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    kgess = float((kge - (-0.41)) / (1.0 - (-0.41)))
    return kge, kgess


def compute_metrics_for_pair(
    sim: np.ndarray,
    obs: np.ndarray,
    eps: float = 1e-6,
    log_eps: float = 0.1,
) -> Dict[str, float]:
    mask = finite_pair_mask(sim, obs)
    n_valid = int(mask.sum())
    row = {
        "n_valid": n_valid,
        "sim_mean": np.nan,
        "obs_mean": np.nan,
        "mean_ratio": np.nan,
        "bias": np.nan,
        "pbias": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "corr": np.nan,
        "nse": np.nan,
        "lognse": np.nan,
        "kge": np.nan,
        "kgess": np.nan,
    }
    if n_valid == 0:
        return row
    sim_valid = sim[mask]
    obs_valid = obs[mask]
    diff = sim_valid - obs_valid
    sim_mean = float(np.mean(sim_valid))
    obs_mean = float(np.mean(obs_valid))
    row["sim_mean"] = sim_mean
    row["obs_mean"] = obs_mean
    if abs(obs_mean) > eps:
        row["mean_ratio"] = float(sim_mean / (obs_mean + eps))
    obs_sum = float(np.sum(obs_valid))
    if abs(obs_sum) > eps:
        row["pbias"] = float(100.0 * np.sum(diff) / (obs_sum + eps))
    row["bias"] = float(np.mean(diff))
    row["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    row["mae"] = float(np.mean(np.abs(diff)))
    row["corr"] = safe_corr(sim_valid, obs_valid, eps=eps)
    row["nse"] = safe_nse(sim_valid, obs_valid, eps=eps)
    row["lognse"] = safe_lognse(sim_valid, obs_valid, log_eps=log_eps, eps=eps)
    row["kge"], row["kgess"] = safe_kge(sim_valid, obs_valid, eps=eps)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LSTM log-residual Qsim post-processor.")
    parser.add_argument(
        "--npz-path",
        type=Path,
        required=True,
        help="Path to station_qsim_qobs_2004_2010.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for LSTM training outputs.",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--target-clip-abs", type=float, default=8.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--station-embedding-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-year", default="2008")
    parser.add_argument("--loss", default="smoothl1", choices=["smoothl1", "mse"])
    parser.add_argument("--smoothl1-beta", type=float, default=0.5)
    parser.add_argument("--use-regime-weights", type=int, choices=[0, 1], default=1)
    parser.add_argument("--regime-quantiles", default="0.1,0.33,0.66,0.9")
    parser.add_argument("--pred-clip-abs", type=float, default=8.0)
    return parser


def parse_val_year(value: str) -> Any:
    if str(value).strip().lower() in {"none", "no", "null", ""}:
        return None
    return int(value)


def parse_regime_quantiles(text: str) -> np.ndarray:
    values = np.array([float(item.strip()) for item in text.split(",") if item.strip()], dtype=np.float64)
    if values.shape[0] != 4:
        raise ValueError("--regime-quantiles must contain exactly four comma-separated values")
    if np.any(values <= 0.0) or np.any(values >= 1.0) or np.any(np.diff(values) <= 0.0):
        raise ValueError("--regime-quantiles must be increasing values between 0 and 1")
    return values


def load_npz_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {npz_path}")
    required = ["dates", "station_ids", "qsim", "qobs", "train_mask_dates", "test_mask_dates"]
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Input NPZ is missing keys: {', '.join(missing)}")
        return {key: data[key] for key in data.files}


def validate_arrays(qsim: np.ndarray, qobs: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> None:
    if qsim.shape != qobs.shape:
        raise ValueError(f"qsim/qobs shape mismatch: {qsim.shape} vs {qobs.shape}")
    if qsim.ndim != 2:
        raise ValueError(f"qsim must be [time, station], got shape {qsim.shape}")
    if train_mask.shape[0] != qsim.shape[0] or test_mask.shape[0] != qsim.shape[0]:
        raise ValueError("train/test mask length must match qsim time dimension")
    finite = qsim[np.isfinite(qsim)]
    if finite.size > 0 and float(np.min(finite)) < -1e-6:
        raise ValueError(f"qsim contains negative values below -1e-6: min={float(np.min(finite))}")


def compute_station_scalers(
    qsim: np.ndarray,
    train_mask: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    time_indices = np.flatnonzero(train_mask)
    time_indices = time_indices[time_indices >= lookback - 1]
    x_mean = np.zeros(qsim.shape[1], dtype=np.float32)
    x_std = np.ones(qsim.shape[1], dtype=np.float32)
    for station_idx in range(qsim.shape[1]):
        values = qsim[time_indices, station_idx] if time_indices.size else np.array([], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            log_values = np.log1p(np.maximum(values, 0.0))
            x_mean[station_idx] = np.float32(np.mean(log_values))
            std = float(np.std(log_values))
            x_std[station_idx] = np.float32(std if std >= 1e-6 else 1.0)
    return x_mean, x_std


def build_sample_indices(
    qsim: np.ndarray,
    qobs: np.ndarray,
    period_mask: np.ndarray,
    lookback: int,
    target_clip_abs: float,
) -> np.ndarray:
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if qsim.shape[0] < lookback:
        return np.zeros((0, 2), dtype=np.int64)

    qsim_valid = np.isfinite(qsim)
    qobs_valid = np.isfinite(qobs)
    csum = np.vstack(
        [
            np.zeros((1, qsim.shape[1]), dtype=np.int32),
            np.cumsum(qsim_valid.astype(np.int32), axis=0),
        ]
    )
    window_counts = csum[lookback : qsim.shape[0] + 1] - csum[0 : qsim.shape[0] - lookback + 1]
    window_valid = np.zeros_like(qsim_valid, dtype=bool)
    window_valid[lookback - 1 :, :] = window_counts == lookback

    with np.errstate(invalid="ignore"):
        target = np.log1p(qobs) - np.log1p(qsim)
    target_valid = np.isfinite(target) & (np.abs(np.clip(target, -target_clip_abs, target_clip_abs)) <= target_clip_abs)
    valid = window_valid & qobs_valid & qsim_valid & target_valid & period_mask[:, None]
    return np.argwhere(valid).astype(np.int64)


def station_regime_thresholds(
    qobs: np.ndarray,
    train_indices: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    thresholds = np.full((qobs.shape[1], 4), np.nan, dtype=np.float32)
    for station_idx in range(qobs.shape[1]):
        mask = train_indices[:, 1] == station_idx
        if not np.any(mask):
            continue
        values = qobs[train_indices[mask, 0], station_idx]
        values = values[np.isfinite(values)]
        if values.size:
            thresholds[station_idx, :] = np.quantile(values, quantiles).astype(np.float32)
    return thresholds


def classify_regimes(qobs: np.ndarray, indices: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    regimes = np.full(indices.shape[0], 2, dtype=np.int64)
    for i, (time_idx, station_idx) in enumerate(indices):
        edges = thresholds[station_idx]
        value = qobs[time_idx, station_idx]
        if np.isfinite(value) and np.all(np.isfinite(edges)):
            regimes[i] = int(np.searchsorted(edges, value, side="right"))
    return regimes


def compute_regime_weights(
    qobs: np.ndarray,
    train_indices: np.ndarray,
    all_indices: Dict[str, np.ndarray],
    quantiles: np.ndarray,
    use_regime_weights: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    thresholds = station_regime_thresholds(qobs, train_indices, quantiles)
    weights_by_regime = np.ones(5, dtype=np.float32)
    train_regimes = classify_regimes(qobs, train_indices, thresholds)
    counts = np.bincount(train_regimes, minlength=5).astype(np.int64)
    if use_regime_weights and train_indices.shape[0] > 0:
        total = float(train_indices.shape[0])
        for regime_idx in range(5):
            if counts[regime_idx] > 0:
                weights_by_regime[regime_idx] = np.float32(total / (5.0 * counts[regime_idx]))
            else:
                weights_by_regime[regime_idx] = np.float32(0.0)
        train_weights_raw = weights_by_regime[train_regimes]
        mean_weight = float(np.mean(train_weights_raw)) if train_weights_raw.size else 1.0
        if mean_weight > 0.0:
            weights_by_regime = weights_by_regime / np.float32(mean_weight)
    else:
        weights_by_regime[:] = 1.0

    sample_weights = {}
    for name, indices in all_indices.items():
        regimes = classify_regimes(qobs, indices, thresholds)
        sample_weights[name] = weights_by_regime[regimes].astype(np.float32)

    rows = []
    for regime_idx, regime_name in enumerate(REGIME_NAMES):
        rows.append(
            {
                "regime": regime_name,
                "count": int(counts[regime_idx]),
                "weight": float(weights_by_regime[regime_idx]),
            }
        )
    return sample_weights, thresholds, pd.DataFrame(rows)


class StationWindowDataset(Dataset):
    def __init__(
        self,
        qsim: np.ndarray,
        qobs: np.ndarray,
        indices: np.ndarray,
        sample_weights: np.ndarray,
        x_mean_by_station: np.ndarray,
        x_std_by_station: np.ndarray,
        lookback: int,
        target_clip_abs: float,
    ) -> None:
        self.qsim = qsim
        self.qobs = qobs
        self.indices = indices.astype(np.int64)
        self.sample_weights = sample_weights.astype(np.float32)
        self.x_mean_by_station = x_mean_by_station.astype(np.float32)
        self.x_std_by_station = x_std_by_station.astype(np.float32)
        self.lookback = int(lookback)
        self.target_clip_abs = float(target_clip_abs)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_idx = int(self.indices[item, 0])
        station_idx = int(self.indices[item, 1])
        window = self.qsim[time_idx - self.lookback + 1 : time_idx + 1, station_idx]
        x = np.log1p(np.maximum(window, 0.0)).astype(np.float32)
        x = (x - self.x_mean_by_station[station_idx]) / self.x_std_by_station[station_idx]
        x = x.reshape(self.lookback, 1)
        qsim_current = np.float32(self.qsim[time_idx, station_idx])
        qobs_current = np.float32(self.qobs[time_idx, station_idx])
        y = np.float32(np.log1p(qobs_current) - np.log1p(qsim_current))
        y = np.float32(np.clip(y, -self.target_clip_abs, self.target_clip_abs))
        return (
            torch.from_numpy(x),
            torch.tensor(station_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(self.sample_weights[item], dtype=torch.float32),
            torch.tensor(qsim_current, dtype=torch.float32),
            torch.tensor(qobs_current, dtype=torch.float32),
            torch.tensor(time_idx, dtype=torch.long),
        )


class LSTMLogResidualCorrector(nn.Module):
    def __init__(
        self,
        num_stations: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        station_embedding_dim: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.use_embedding = station_embedding_dim > 0
        if self.use_embedding:
            self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
            head_in = hidden_size + station_embedding_dim
        else:
            self.station_embedding = None
            head_in = hidden_size
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, station_index: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        if self.use_embedding:
            emb = self.station_embedding(station_index)
            features = torch.cat([last_hidden, emb], dim=1)
        else:
            features = last_hidden
        return self.head(features).squeeze(-1)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def loss_per_sample(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.loss == "smoothl1":
        return nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=args.smoothl1_beta)
    return nn.functional.mse_loss(pred, target, reduction="none")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Any = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_weight = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x, station_idx, y, weight, _, _, _ = batch
            x = x.to(device, non_blocking=True)
            station_idx = station_idx.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x, station_idx)
            per_sample = loss_per_sample(pred, y, args)
            loss = torch.sum(per_sample * weight) / torch.clamp(torch.sum(weight), min=1.0)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(torch.sum(per_sample.detach() * weight).cpu())
            total_weight += float(torch.sum(weight).cpu())
    if total_weight <= 0:
        return np.nan
    return total_loss / total_weight


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    station_ids: np.ndarray,
    x_mean_by_station: np.ndarray,
    x_std_by_station: np.ndarray,
    best_val_loss: float,
    epoch: int,
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "station_ids": station_ids,
        "x_mean_by_station": x_mean_by_station,
        "x_std_by_station": x_std_by_station,
        "target_clip_abs": args.target_clip_abs,
        "lookback": args.lookback,
        "best_val_loss": best_val_loss,
        "epoch": epoch,
    }


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    # This checkpoint is generated by this local training script, so
    # weights_only=False is required to load numpy metadata stored in it.
    return torch.load(path, map_location=device, weights_only=False)


def predict_to_matrix(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    qsim: np.ndarray,
    qobs: np.ndarray,
    device: torch.device,
    pred_clip_abs: float,
) -> np.ndarray:
    qcorr = np.full_like(qsim, np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for _, loader in loaders.items():
            for batch in loader:
                x, station_idx, _, _, qsim_current, _, time_idx = batch
                x = x.to(device, non_blocking=True)
                station_idx_device = station_idx.to(device, non_blocking=True)
                pred = model(x, station_idx_device)
                pred = torch.clamp(pred, -pred_clip_abs, pred_clip_abs).cpu().numpy()
                qsim_np = qsim_current.numpy()
                qcorr_values = np.expm1(np.log1p(np.maximum(qsim_np, 0.0)) + pred)
                qcorr_values = np.maximum(qcorr_values, 0.0).astype(np.float32)
                qcorr[time_idx.numpy(), station_idx.numpy()] = qcorr_values
    return qcorr


def compute_period_metrics(
    qsim: np.ndarray,
    qobs: np.ndarray,
    qcorr: np.ndarray,
    station_ids: np.ndarray,
    period_masks: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for period_name, period_mask in period_masks.items():
        for station_idx, station_id in enumerate(station_ids):
            eval_mask = period_mask & np.isfinite(qcorr[:, station_idx]) & np.isfinite(qobs[:, station_idx])
            raw_metrics = compute_metrics_for_pair(qsim[eval_mask, station_idx], qobs[eval_mask, station_idx])
            lstm_metrics = compute_metrics_for_pair(qcorr[eval_mask, station_idx], qobs[eval_mask, station_idx])
            for model_name, metrics in [("raw", raw_metrics), ("lstm", lstm_metrics)]:
                row = {
                    "station_id": int(station_id),
                    "period": period_name,
                    "model": model_name,
                }
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, model), group in metrics_df.groupby(["period", "model"], sort=False):
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row = {
                "period": period,
                "model": model,
                "metric": metric,
                "count": int(finite.size),
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "max": np.nan,
            }
            if finite.size:
                row["mean"] = float(np.mean(finite))
                row["median"] = float(np.median(finite))
                row["std"] = float(np.std(finite))
                row["min"] = float(np.min(finite))
                row["p25"] = float(np.percentile(finite, 25))
                row["p75"] = float(np.percentile(finite, 75))
                row["max"] = float(np.max(finite))
            rows.append(row)
    return pd.DataFrame(rows)


def metric_median(metrics_df: pd.DataFrame, period: str, model: str, metric: str) -> float:
    values = metrics_df.loc[
        (metrics_df["period"] == period) & (metrics_df["model"] == model), metric
    ].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return np.nan if values.size == 0 else float(np.median(values))


def compute_skill_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ["train", "val", "test"]:
        raw = metrics_df[(metrics_df["period"] == period) & (metrics_df["model"] == "raw")]
        lstm = metrics_df[(metrics_df["period"] == period) & (metrics_df["model"] == "lstm")]
        merged = raw.merge(lstm, on=["station_id", "period"], suffixes=("_raw", "_lstm"))
        row = {"period": period}
        for metric in ["corr", "rmse", "nse", "kge"]:
            raw_median = metric_median(metrics_df, period, "raw", metric)
            lstm_median = metric_median(metrics_df, period, "lstm", metric)
            row[f"median_raw_{metric}"] = raw_median
            row[f"median_lstm_{metric}"] = lstm_median
            row[f"median_delta_lstm_minus_raw_{metric}"] = (
                np.nan if not (np.isfinite(raw_median) and np.isfinite(lstm_median)) else lstm_median - raw_median
            )
        if merged.empty:
            row["stations_improved_corr"] = 0
            row["stations_improved_rmse"] = 0
            row["stations_improved_nse"] = 0
            row["stations_improved_kge"] = 0
        else:
            row["stations_improved_corr"] = int((merged["corr_lstm"] > merged["corr_raw"]).sum())
            row["stations_improved_rmse"] = int((merged["rmse_lstm"] < merged["rmse_raw"]).sum())
            row["stations_improved_nse"] = int((merged["nse_lstm"] > merged["nse_raw"]).sum())
            row["stations_improved_kge"] = int((merged["kge_lstm"] > merged["kge_raw"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)

    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lookback < 1:
        raise ValueError("--lookback must be >= 1")
    val_year = parse_val_year(args.val_year)
    regime_quantiles = parse_regime_quantiles(args.regime_quantiles)


    data = load_npz_dataset(npz_path)
    dates = pd.DatetimeIndex(pd.to_datetime(data["dates"].astype(str)))
    station_ids = data["station_ids"].astype(np.int64)
    qsim = data["qsim"].astype(np.float32)
    qobs = data["qobs"].astype(np.float32)
    train_mask_dates = data["train_mask_dates"].astype(bool)
    test_mask_dates = data["test_mask_dates"].astype(bool)
    validate_arrays(qsim, qobs, train_mask_dates, test_mask_dates)

    qsim = np.where(np.isfinite(qsim), np.maximum(qsim, 0.0), qsim).astype(np.float32)


    train_period_mask = train_mask_dates.copy()
    if val_year is None:
        val_period_mask = np.zeros_like(train_mask_dates, dtype=bool)
    else:
        val_period_mask = train_mask_dates & (dates.year == val_year)
        train_period_mask = train_mask_dates & (dates.year != val_year)
    period_masks = {
        "train": train_period_mask,
        "val": val_period_mask,
        "test": test_mask_dates,
    }

    train_indices = build_sample_indices(qsim, qobs, train_period_mask, args.lookback, args.target_clip_abs)
    val_indices = build_sample_indices(qsim, qobs, val_period_mask, args.lookback, args.target_clip_abs)
    test_indices = build_sample_indices(qsim, qobs, test_mask_dates, args.lookback, args.target_clip_abs)

    if train_indices.shape[0] == 0:
        raise ValueError("No train samples were constructed.")

    print(f"[SAMPLES] train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}", flush=True)

    x_mean, x_std = compute_station_scalers(qsim, train_period_mask, args.lookback)
    scalers_path = out_dir / "scalers_logqsim_by_station.npz"
    np.savez_compressed(
        scalers_path,
        station_ids=station_ids,
        x_mean_by_station=x_mean,
        x_std_by_station=x_std,
        lookback=np.array(args.lookback, dtype=np.int64),
        target_clip_abs=np.array(args.target_clip_abs, dtype=np.float32),
    )

    all_indices = {"train": train_indices, "val": val_indices, "test": test_indices}
    sample_weights, regime_thresholds, regime_df = compute_regime_weights(
        qobs=qobs,
        train_indices=train_indices,
        all_indices=all_indices,
        quantiles=regime_quantiles,
        use_regime_weights=bool(args.use_regime_weights),
    )
    regime_df.to_csv(out_dir / "regime_weights_summary.csv", index=False)
    np.savez_compressed(out_dir / "regime_thresholds_by_station.npz", station_ids=station_ids, thresholds=regime_thresholds)

    train_dataset = StationWindowDataset(qsim, qobs, train_indices, sample_weights["train"], x_mean, x_std, args.lookback, args.target_clip_abs)
    val_dataset = StationWindowDataset(qsim, qobs, val_indices, sample_weights["val"], x_mean, x_std, args.lookback, args.target_clip_abs)
    test_dataset = StationWindowDataset(qsim, qobs, test_indices, sample_weights["test"], x_mean, x_std, args.lookback, args.target_clip_abs)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        requested_device = "cpu"
    device = torch.device(requested_device)
    pin_memory = device.type == "cuda"
    print(f"[DEVICE] requested={args.device}, using={device}", flush=True)
    print(f"[DEVICE] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DEVICE] gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_loader = make_dataloader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_dataloader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_dataloader(test_dataset, args.batch_size, False, args.num_workers, pin_memory)

    model = LSTMLogResidualCorrector(
        num_stations=len(station_ids),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        station_embedding_dim=args.station_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_model_path = out_dir / "best_model.pt"
    final_model_path = out_dir / "final_model.pt"
    history_rows = []
    best_val_loss = np.inf
    best_epoch = -1
    epochs_without_improve = 0
    use_early_stopping = val_year is not None and len(val_dataset) > 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, args, optimizer=optimizer)
        val_loss = np.nan
        if len(val_dataset) > 0:
            val_loss = run_epoch(model, val_loader, device, args, optimizer=None)
            scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr_now,
            }
        )
        print(
            f"[EPOCH] epoch={epoch} train_loss={train_loss:.6g} val_loss={val_loss:.6g} "
            f"lr={lr_now:.6g}",
            flush=True,
        )
        if use_early_stopping and np.isfinite(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(
                    checkpoint_payload(model, optimizer, args, station_ids, x_mean, x_std, best_val_loss, epoch),
                    best_model_path,
                )
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.patience:
                    print(f"[EARLY_STOP] patience reached at epoch {epoch}", flush=True)
                    break

    if not use_early_stopping:
        best_val_loss = np.nan
        best_epoch = len(history_rows)

    torch.save(
        checkpoint_payload(model, optimizer, args, station_ids, x_mean, x_std, best_val_loss, best_epoch),
        final_model_path,
    )
    if use_early_stopping and best_model_path.exists():
        checkpoint = load_checkpoint(best_model_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif not use_early_stopping:
        best_model_path = final_model_path

    history_path = out_dir / "train_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    loaders_for_prediction = {"train": train_loader, "val": val_loader, "test": test_loader}
    qcorr = predict_to_matrix(model, loaders_for_prediction, qsim, qobs, device, args.pred_clip_abs)
    qcorr_path = out_dir / "lstm_qcorr_2004_2010.npz"
    np.savez_compressed(
        qcorr_path,
        dates=np.array([date.strftime("%Y-%m-%d") for date in dates], dtype="U10"),
        station_ids=station_ids,
        qsim=qsim.astype(np.float32),
        qobs=qobs.astype(np.float32),
        qcorr=qcorr.astype(np.float32),
        train_mask_dates=train_mask_dates.astype(bool),
        test_mask_dates=test_mask_dates.astype(bool),
        lookback=np.array(args.lookback, dtype=np.int64),
        target_mode=np.array("log_residual"),
        target_clip_abs=np.array(args.target_clip_abs, dtype=np.float32),
        pred_clip_abs=np.array(args.pred_clip_abs, dtype=np.float32),
    )

    metrics_df = compute_period_metrics(qsim, qobs, qcorr, station_ids, period_masks)
    metrics_summary_df = summarize_metrics(metrics_df)
    skill_df = compute_skill_summary(metrics_df)
    metrics_path = out_dir / "lstm_metrics_by_station_period.csv"
    metrics_summary_path = out_dir / "lstm_metrics_summary.csv"
    skill_path = out_dir / "lstm_vs_raw_skill_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    metrics_summary_df.to_csv(metrics_summary_path, index=False)
    skill_df.to_csv(skill_path, index=False)

    test_raw_corr = metric_median(metrics_df, "test", "raw", "corr")
    test_lstm_corr = metric_median(metrics_df, "test", "lstm", "corr")
    test_raw_rmse = metric_median(metrics_df, "test", "raw", "rmse")
    test_lstm_rmse = metric_median(metrics_df, "test", "lstm", "rmse")
    test_raw_nse = metric_median(metrics_df, "test", "raw", "nse")
    test_lstm_nse = metric_median(metrics_df, "test", "lstm", "nse")
    test_raw_kge = metric_median(metrics_df, "test", "raw", "kge")
    test_lstm_kge = metric_median(metrics_df, "test", "lstm", "kge")

    summary_path = out_dir / "lstm_training_summary.txt"
    lines = [
        "LSTM log-residual post-processing training summary",
        "",
        f"npz_path: {npz_path}",
        f"out_dir: {out_dir}",
        "",
        f"station count: {len(station_ids)}",
        f"date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}",
        f"lookback: {args.lookback}",
        "target mode: log_residual",
        f"target clip: {args.target_clip_abs}",
        f"pred clip: {args.pred_clip_abs}",
        f"train samples: {len(train_indices)}",
        f"val samples: {len(val_indices)}",
        f"test samples: {len(test_indices)}",
        "",
        f"hidden_size: {args.hidden_size}",
        f"num_layers: {args.num_layers}",
        f"dropout: {args.dropout}",
        f"station_embedding_dim: {args.station_embedding_dim}",
        f"batch_size: {args.batch_size}",
        f"epochs_requested: {args.epochs}",
        f"lr: {args.lr}",
        f"weight_decay: {args.weight_decay}",
        f"loss: {args.loss}",
        f"smoothl1_beta: {args.smoothl1_beta}",
        "",
        f"best epoch: {best_epoch}",
        f"best val loss: {best_val_loss}",
        "",
        f"test raw median corr/rmse/nse/kge: {test_raw_corr} / {test_raw_rmse} / {test_raw_nse} / {test_raw_kge}",
        f"test LSTM median corr/rmse/nse/kge: {test_lstm_corr} / {test_lstm_rmse} / {test_lstm_nse} / {test_lstm_kge}",
        "",
        f"best_model: {best_model_path}",
        f"final_model: {final_model_path}",
        f"train_history: {history_path}",
        f"scalers: {scalers_path}",
        f"regime_weights_summary: {out_dir / 'regime_weights_summary.csv'}",
        f"qcorr_npz: {qcorr_path}",
        f"metrics_by_station_period: {metrics_path}",
        f"metrics_summary: {metrics_summary_path}",
        f"skill_summary: {skill_path}",
        f"training_summary: {summary_path}",
        "",
        "This is an offline LSTM post-processing workflow. It does not modify CaMa physical states and does not shift the original time axis.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] summary written: {summary_path}", flush=True)
    print("This is an offline LSTM post-processing workflow. It does not modify CaMa physical states and does not shift the original time axis.", flush=True)


if __name__ == "__main__":
    main()
