#!/usr/bin/env python
"""Build US 5 arcmin station Qsim/Qobs arrays for LSTM post-processing.

It uses the zero-based grid indices supplied by the US 5 arcmin allocation CSV,
loads each compressed annual CaMa outflow cube exactly once, shifts CaMa time by
one day, and computes Jiang-form baseline metrics from the generated NPZ.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr



CAMA_OUT_DIR = "./data/cama_output"
MAPPING_CSV = "./data/US_5arcmin_station_mapping.csv"

GRDC_DIR = "./GRDC"
OUT_DIR = "./out"

DATA_START = "2004-01-01"
DATA_END = "2010-12-31"
TRAIN_START = "2004-01-01"
TRAIN_END = "2008-12-31"
TEST_START = "2009-01-01"
TEST_END = "2010-12-31"
LOOKBACK = 60
VALID_RATIO_THRESHOLD = 0.8
MIN_TRAIN_SAMPLES = 365
MIN_TEST_SAMPLES = 100
MAX_NEGATIVE_QSIM_RATIO = 0.05

NX = 708
NY = 384
QSIM_SCALE = 1.0
QSIM_SOURCE_UNITS = "m3/s"
QOBS_UNITS = "m3/s"
TIME_SHIFT_DAYS = -1
COORD_TOLERANCE = 1.0e-4


REQUIRED_MAPPING_COLUMNS = [
    "station_id",
    "lat",
    "lon",
    "grid_lat",
    "grid_lon",
    "x_index",
    "y_index",
]


class BuildLogger:
    """Mirror concise build messages to the terminal and an OUT_DIR log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def station_id_to_str(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("station_id contains a missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("station_id contains an empty value")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"station_id must be an integer-like GRDC ID, got {value!r}")
    return str(int(number))


def station_ids_to_int64(values: Sequence[Any]) -> np.ndarray:
    output: List[int] = []
    for value in values:
        text = station_id_to_str(value)
        try:
            output.append(int(text))
        except ValueError as exc:
            raise ValueError(
                f"station_id {value!r} is not numeric; current LSTM readers require int64 IDs"
            ) from exc
    return np.asarray(output, dtype=np.int64)


def read_mapping_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"5 arcmin station mapping CSV not found: {path}")

    mapping = pd.read_csv(path, dtype={"station_id": str})
    missing = [column for column in REQUIRED_MAPPING_COLUMNS if column not in mapping.columns]
    if missing:
        raise ValueError(
            f"Mapping CSV {path} is missing required columns: {', '.join(missing)}"
        )
    if mapping.empty:
        raise ValueError(f"Mapping CSV contains no stations: {path}")

    mapping = mapping[REQUIRED_MAPPING_COLUMNS].copy()
    mapping["station_id"] = mapping["station_id"].map(station_id_to_str)
    duplicated = mapping["station_id"].duplicated(keep=False)
    if duplicated.any():
        examples = mapping.loc[duplicated, "station_id"].head(10).tolist()
        raise ValueError(f"station_id must be unique; duplicate examples: {examples}")

    numeric_columns = ["lat", "lon", "grid_lat", "grid_lon", "x_index", "y_index"]
    for column in numeric_columns:
        mapping[column] = pd.to_numeric(mapping[column], errors="raise")
        values = mapping[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Mapping column {column!r} contains non-finite values")

    for column in ["x_index", "y_index"]:
        values = mapping[column].to_numpy(dtype=np.float64)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Mapping column {column!r} must contain integer zero-based indices")
        mapping[column] = values.astype(np.int64)

    bad_x = (mapping["x_index"] < 0) | (mapping["x_index"] >= NX)
    bad_y = (mapping["y_index"] < 0) | (mapping["y_index"] >= NY)
    if bad_x.any() or bad_y.any():
        bad = mapping.loc[bad_x | bad_y, ["station_id", "x_index", "y_index"]].head(10)
        raise IndexError(
            f"Mapping indices must satisfy 0 <= x_index < {NX} and 0 <= y_index < {NY}; "
            f"bad examples: {bad.to_dict(orient='records')}"
        )

    if ((mapping["lat"] < -90.0) | (mapping["lat"] > 90.0)).any():
        raise ValueError("Mapping station lat must be within [-90, 90]")
    if ((mapping["grid_lat"] < -90.0) | (mapping["grid_lat"] > 90.0)).any():
        raise ValueError("Mapping grid_lat must be within [-90, 90]")

    # Validate integer compatibility now so failures occur before any annual cube is loaded.
    station_ids_to_int64(mapping["station_id"].tolist())
    return mapping.reset_index(drop=True)


def find_yearly_outflw_files(cama_out_dir: Path, years: Sequence[int]) -> Dict[int, Path]:
    if not cama_out_dir.is_dir():
        raise NotADirectoryError(f"CaMa outflow directory not found: {cama_out_dir}")
    files: Dict[int, Path] = {}
    problems: List[str] = []
    for year in years:
        expected_name = f"o_outflw{year}.nc"
        matches = sorted(path for path in cama_out_dir.rglob(expected_name) if path.is_file())
        if len(matches) == 1:
            files[year] = matches[0]
        elif not matches:
            problems.append(f"{year}: missing {expected_name} under {cama_out_dir}")
        else:
            problems.append(
                f"{year}: expected exactly one {expected_name}, found {len(matches)}: "
                + "; ".join(str(path) for path in matches)
            )
    if problems:
        raise FileNotFoundError(
            "CaMa annual outflow inventory is missing or non-unique:\n" + "\n".join(problems)
        )
    return files


def unit_is_m3_per_second(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return any(unit_is_m3_per_second(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return any(unit_is_m3_per_second(item) for item in value)
    if value is None:
        return False
    text = str(value).strip().lower().replace("³", "3")
    text = text.replace("⁻¹", "-1").replace("−", "-")
    text = text.replace("cubic meters", "m3").replace("cubic metres", "m3")
    text = text.replace("seconds", "s").replace("second", "s")
    text = text.replace("secs", "s").replace("sec", "s")
    compact = re.sub(r"[\s\^*{}()_\[\]',\"]", "", text)
    accepted_fragments = ("m3/s", "m3s-1", "m3s−1")
    return any(fragment in compact for fragment in accepted_fragments)


def require_m3_per_second(units: Any, label: str) -> None:
    if not unit_is_m3_per_second(units):
        raise ValueError(
            f"{label} units must explicitly be m3/s; got {units!r}. "
            "No implicit daily-volume conversion is allowed."
        )


def validate_time_index(index: pd.DatetimeIndex, label: str) -> None:
    if len(index) == 0:
        raise ValueError(f"{label} has an empty time coordinate")
    if index.isna().any():
        raise ValueError(f"{label} has unparseable/NaT time values")
    if index.has_duplicates:
        duplicates = index[index.duplicated(keep=False)][:10]
        raise ValueError(f"{label} has duplicate dates, examples={duplicates.tolist()}")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label} time is not strictly monotonic increasing")


def extract_qsim_station_matrix(
    cama_out_dir: Path,
    mapping: pd.DataFrame,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    logger: BuildLogger,
) -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, Any]]:
    years = list(range(data_start.year, data_end.year + 1))
    yearly_files = find_yearly_outflw_files(cama_out_dir, years)
    x_indices = mapping["x_index"].to_numpy(dtype=np.int64)
    y_indices = mapping["y_index"].to_numpy(dtype=np.int64)
    grid_lon = mapping["grid_lon"].to_numpy(dtype=np.float64)
    grid_lat = mapping["grid_lat"].to_numpy(dtype=np.float64)

    corrected_times: List[pd.DatetimeIndex] = []
    station_arrays: List[np.ndarray] = []
    yearly_info: List[Dict[str, Any]] = []
    grid_lon_template: Optional[np.ndarray] = None
    grid_lat_template: Optional[np.ndarray] = None

    for year in years:
        path = yearly_files[year]
        with xr.open_dataset(path) as ds:
            if "outflw" not in ds.variables:
                raise ValueError(f"{path} does not contain variable 'outflw'")
            da = ds["outflw"]
            if len(da.dims) != 3 or set(da.dims) != {"time", "lat", "lon"}:
                raise ValueError(
                    f"{path} outflw must have exactly dimensions time/lat/lon; got {da.dims}"
                )
            if "time" not in da.coords or "lat" not in da.coords or "lon" not in da.coords:
                raise ValueError(f"{path} outflw must expose time, lat, and lon coordinates")
            if da.sizes["lon"] != NX or da.sizes["lat"] != NY:
                raise ValueError(
                    f"{path} grid must be lat={NY}, lon={NX}; got "
                    f"lat={da.sizes['lat']}, lon={da.sizes['lon']}"
                )
            if da["lon"].dims != ("lon",) or da["lat"].dims != ("lat",):
                raise ValueError(f"{path} lon and lat coordinates must both be one-dimensional")
            require_m3_per_second(da.attrs.get("units"), f"{path}:outflw")

            lon_values = np.asarray(da["lon"].values, dtype=np.float64)
            lat_values = np.asarray(da["lat"].values, dtype=np.float64)
            if not np.isfinite(lon_values).all() or not np.isfinite(lat_values).all():
                raise ValueError(f"{path} contains non-finite lon/lat coordinates")
            selected_lon = lon_values[x_indices]
            selected_lat = lat_values[y_indices]
            lon_error = np.abs(selected_lon - grid_lon)
            lat_error = np.abs(selected_lat - grid_lat)
            if np.any(lon_error >= COORD_TOLERANCE) or np.any(lat_error >= COORD_TOLERANCE):
                bad_mask = (lon_error >= COORD_TOLERANCE) | (lat_error >= COORD_TOLERANCE)
                bad_pos = np.flatnonzero(bad_mask)[:10]
                examples = [
                    {
                        "station_id": mapping.iloc[pos]["station_id"],
                        "x_index": int(x_indices[pos]),
                        "y_index": int(y_indices[pos]),
                        "csv_grid_lon": float(grid_lon[pos]),
                        "nc_lon": float(selected_lon[pos]),
                        "csv_grid_lat": float(grid_lat[pos]),
                        "nc_lat": float(selected_lat[pos]),
                    }
                    for pos in bad_pos
                ]
                raise ValueError(
                    f"{path} mapping-to-NetCDF coordinate check failed at tolerance "
                    f"{COORD_TOLERANCE}: {examples}"
                )
            if grid_lon_template is None:
                grid_lon_template = selected_lon.copy()
                grid_lat_template = selected_lat.copy()
            elif not (
                np.allclose(selected_lon, grid_lon_template, rtol=0.0, atol=COORD_TOLERANCE)
                and np.allclose(selected_lat, grid_lat_template, rtol=0.0, atol=COORD_TOLERANCE)
            ):
                raise ValueError(f"{path} station grid coordinates differ from the first annual file")

            original_time = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
            validate_time_index(original_time, f"{path}:original_time")
            corrected_time = original_time + pd.Timedelta(days=TIME_SHIFT_DAYS)
            validate_time_index(corrected_time, f"{path}:corrected_time")

            # Critical performance path: decompress and load this complete annual cube once,
            # then use NumPy point indexing in memory.
            transposed = da.transpose("time", "lat", "lon")
            raw = transposed.astype("float32").load().values
            if raw.shape != (len(original_time), NY, NX):
                raise ValueError(
                    f"Unexpected loaded shape for {path}: {raw.shape}; "
                    f"expected {(len(original_time), NY, NX)}"
                )
            station_values = raw[:, y_indices, x_indices]
            if station_values.shape != (len(original_time), len(mapping)):
                raise ValueError(
                    f"Unexpected extracted station shape for {path}: {station_values.shape}"
                )

            logger.log(f"[QSIM {year}] file: {path}")
            logger.log(
                f"[QSIM {year}] original dates: "
                f"{original_time[0].date()} .. {original_time[-1].date()}"
            )
            logger.log(
                f"[QSIM {year}] shifted dates ({TIME_SHIFT_DAYS} day): "
                f"{corrected_time[0].date()} .. {corrected_time[-1].date()}"
            )
            logger.log(
                f"[QSIM {year}] raw shape: {raw.shape}; "
                f"station matrix shape: {station_values.shape}"
            )

            yearly_info.append(
                {
                    "year": year,
                    "path": str(path),
                    "units": str(da.attrs.get("units")),
                    "original_start": original_time[0].strftime("%Y-%m-%d"),
                    "original_end": original_time[-1].strftime("%Y-%m-%d"),
                    "corrected_start": corrected_time[0].strftime("%Y-%m-%d"),
                    "corrected_end": corrected_time[-1].strftime("%Y-%m-%d"),
                    "raw_shape": tuple(raw.shape),
                    "station_shape": tuple(station_values.shape),
                }
            )
            # corrected_time is saved before annual concatenation by construction.
            corrected_times.append(corrected_time)
            station_arrays.append(np.asarray(station_values, dtype=np.float32))

    all_dates = pd.DatetimeIndex(np.concatenate([index.values for index in corrected_times]))
    qsim_all_dates = np.concatenate(station_arrays, axis=0).astype(np.float32, copy=False)
    validate_time_index(all_dates, "merged corrected CaMa time")

    in_period = (all_dates >= data_start) & (all_dates <= data_end)
    dates = all_dates[in_period]
    qsim = qsim_all_dates[np.asarray(in_period), :]
    expected_dates = pd.date_range(data_start, data_end, freq="D")
    if not dates.equals(expected_dates):
        missing = expected_dates.difference(dates)
        unexpected = dates.difference(expected_dates)
        raise ValueError(
            "Corrected/cropped CaMa dates are not a complete daily sequence: "
            f"actual={len(dates)}, expected={len(expected_dates)}, "
            f"first={dates[0] if len(dates) else None}, last={dates[-1] if len(dates) else None}, "
            f"missing_examples={missing[:10].tolist()}, "
            f"unexpected_examples={unexpected[:10].tolist()}"
        )
    if data_start == pd.Timestamp(DATA_START) and data_end == pd.Timestamp(DATA_END):
        if len(dates) != 2557 or dates[0] != pd.Timestamp(DATA_START) or dates[-1] != pd.Timestamp(DATA_END):
            raise ValueError(
                "Default build must contain exactly 2557 dates from 2004-01-01 through 2010-12-31"
            )

    info = {
        "yearly_info": yearly_info,
        "lat_cama": np.asarray(grid_lat_template, dtype=np.float64),
        "lon_cama": np.asarray(grid_lon_template, dtype=np.float64),
        "extraction_method": "full_annual_cube_load_once_then_numpy_point_index",
    }
    return dates, qsim, info


def load_qobs_matrix(
    mapping: pd.DataFrame,
    grdc_dir: Path,
    dates: pd.DatetimeIndex,
    logger: BuildLogger,
) -> Tuple[np.ndarray, np.ndarray]:
    if not grdc_dir.is_dir():
        raise NotADirectoryError(f"GRDC directory not found: {grdc_dir}")
    station_ids = mapping["station_id"].tolist()
    paths = [grdc_dir / f"{station_id}_Q_Day.Cmd.nc" for station_id in station_ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} mapped GRDC station files; examples:\n"
            + "\n".join(missing[:20])
        )

    qobs = np.full((len(dates), len(mapping)), np.nan, dtype=np.float32)
    covers_period = np.zeros(len(mapping), dtype=bool)
    logger.log(f"[QOBS] loading {len(mapping)} GRDC station files")
    for station_index, (station_id, path) in enumerate(zip(station_ids, paths)):
        with xr.open_dataset(path) as ds:
            if "discharge" not in ds.variables:
                raise ValueError(f"GRDC file has no discharge variable: {path}")
            discharge = ds["discharge"].squeeze(drop=True)
            if discharge.dims != ("time",):
                raise ValueError(
                    f"GRDC discharge must reduce to exactly dimension ('time',) in {path}; "
                    f"got {discharge.dims}"
                )
            if "time" not in discharge.coords:
                raise ValueError(f"GRDC discharge has no time coordinate: {path}")
            require_m3_per_second(discharge.attrs.get("units"), f"{path}:discharge")
            obs_dates = pd.DatetimeIndex(pd.to_datetime(discharge["time"].values))
            validate_time_index(obs_dates, f"{path}:time")
            values = np.asarray(discharge.values, dtype=np.float64)
            if values.shape != (len(obs_dates),):
                raise ValueError(f"Unexpected discharge shape in {path}: {values.shape}")
            values[values <= -900.0] = np.nan
            series = pd.Series(values, index=obs_dates)
            qobs[:, station_index] = series.reindex(dates).to_numpy(dtype=np.float32)
            covers_period[station_index] = bool(
                obs_dates[0] <= dates[0] and obs_dates[-1] >= dates[-1]
            )
        if (station_index + 1) % 100 == 0 or station_index + 1 == len(mapping):
            logger.log(f"[QOBS] loaded {station_index + 1}/{len(mapping)} stations")
    return qobs, covers_period


def compute_lstm_valid_samples(qsim: np.ndarray, qobs: np.ndarray, lookback: int) -> np.ndarray:
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if qsim.shape != qobs.shape or qsim.ndim != 2:
        raise ValueError(f"qsim/qobs must be matching 2-D arrays; got {qsim.shape}, {qobs.shape}")
    samples = np.zeros(qsim.shape, dtype=bool)
    if qsim.shape[0] < lookback:
        return samples
    qsim_valid = np.isfinite(qsim)
    target_valid = qsim_valid & np.isfinite(qobs)
    cumulative = np.vstack(
        [
            np.zeros((1, qsim.shape[1]), dtype=np.int32),
            np.cumsum(qsim_valid.astype(np.int32), axis=0),
        ]
    )
    window_counts = cumulative[lookback:] - cumulative[:-lookback]
    samples[lookback - 1 :, :] = (
        target_valid[lookback - 1 :, :] & (window_counts == lookback)
    )
    return samples


def masked_column_mean(values: np.ndarray, date_mask: np.ndarray) -> np.ndarray:
    selected = values[np.asarray(date_mask), :]
    finite = np.isfinite(selected)
    counts = finite.sum(axis=0)
    sums = np.where(finite, selected, 0.0).sum(axis=0, dtype=np.float64)
    output = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


def finite_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def jiang_metrics(sim: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(sim) & np.isfinite(obs)
    result: Dict[str, Any] = {
        "valid_days": int(valid.sum()),
        "CC": np.nan,
        "BR": np.nan,
        "RV": np.nan,
        "KGE": np.nan,
        "qsim_mean": np.nan,
        "qobs_mean": np.nan,
        "qsim_std": np.nan,
        "qobs_std": np.nan,
    }
    if result["valid_days"] < 2:
        return result
    sim_valid = np.asarray(sim[valid], dtype=np.float64)
    obs_valid = np.asarray(obs[valid], dtype=np.float64)
    sim_mean = float(np.mean(sim_valid))
    obs_mean = float(np.mean(obs_valid))
    sim_std = float(np.std(sim_valid, ddof=0))
    obs_std = float(np.std(obs_valid, ddof=0))
    result.update(
        {
            "qsim_mean": sim_mean,
            "qobs_mean": obs_mean,
            "qsim_std": sim_std,
            "qobs_std": obs_std,
        }
    )
    eps = 1.0e-12
    if abs(sim_mean) <= eps or abs(obs_mean) <= eps or sim_std <= eps or obs_std <= eps:
        return result
    cc = float(np.corrcoef(sim_valid, obs_valid)[0, 1])
    br = float(sim_mean / obs_mean)
    rv = float((sim_std / sim_mean) / (obs_std / obs_mean))
    if np.isfinite(cc) and np.isfinite(br) and np.isfinite(rv):
        kge = float(1.0 - math.sqrt((cc - 1.0) ** 2 + (br - 1.0) ** 2 + (rv - 1.0) ** 2))
    else:
        kge = np.nan
    result.update({"CC": cc, "BR": br, "RV": rv, "KGE": kge})
    return result


def compute_baseline_from_npz(
    npz_path: Path,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    with np.load(npz_path, allow_pickle=False) as data:
        required = ["dates", "station_ids", "qsim", "qobs"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Cannot compute baseline; NPZ is missing keys: {missing}")
        dates = pd.DatetimeIndex(pd.to_datetime(data["dates"]))
        station_ids = np.asarray(data["station_ids"], dtype=np.int64)
        qsim = np.asarray(data["qsim"], dtype=np.float32)
        qobs = np.asarray(data["qobs"], dtype=np.float32)
    expected_dates = pd.date_range(period_start, period_end, freq="D")
    if not dates.equals(expected_dates):
        raise ValueError(
            f"Baseline verification requires NPZ dates {period_start.date()}..{period_end.date()}; "
            f"got {dates[0] if len(dates) else None}..{dates[-1] if len(dates) else None}"
        )
    if qsim.shape != qobs.shape or qsim.shape != (len(dates), len(station_ids)):
        raise ValueError(
            f"NPZ baseline shapes are inconsistent: qsim={qsim.shape}, qobs={qobs.shape}, "
            f"dates={len(dates)}, stations={len(station_ids)}"
        )

    rows: List[Dict[str, Any]] = []
    for station_index, station_id in enumerate(station_ids):
        row: Dict[str, Any] = {"station_id": int(station_id)}
        row.update(jiang_metrics(qsim[:, station_index], qobs[:, station_index]))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    valid_metrics = metrics.loc[np.isfinite(metrics["KGE"])].copy()
    if valid_metrics.empty:
        raise ValueError("No stations have finite Jiang CC/BR/RV/KGE metrics")
    kge = valid_metrics["KGE"].to_numpy(dtype=np.float64)
    summary = {
        "valid_station_count": float(len(valid_metrics)),
        "median_KGE": float(np.median(kge)),
        "mean_KGE": float(np.mean(kge)),
        "KGE_gt0_ratio": float(np.mean(kge > 0.0)),
        "KGE_gt05_ratio": float(np.mean(kge > 0.5)),
        "median_CC": float(np.median(valid_metrics["CC"])),
        "median_BR": float(np.median(valid_metrics["BR"])),
        "median_RV": float(np.median(valid_metrics["RV"])),
    }
    return metrics, summary




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ERA5-Land US 5 arcmin station Qsim/Qobs arrays for LSTM."
    )
    parser.add_argument("--cama-out-dir", default=CAMA_OUT_DIR)
    parser.add_argument("--mapping-csv", default=MAPPING_CSV)
    parser.add_argument("--grdc-dir", default=GRDC_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--data-start", default=DATA_START)
    parser.add_argument("--data-end", default=DATA_END)
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--test-start", default=TEST_START)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--valid-ratio-threshold", type=float, default=VALID_RATIO_THRESHOLD)
    parser.add_argument("--min-train-samples", type=int, default=MIN_TRAIN_SAMPLES)
    parser.add_argument("--min-test-samples", type=int, default=MIN_TEST_SAMPLES)
    parser.add_argument("--max-negative-qsim-ratio", type=float, default=MAX_NEGATIVE_QSIM_RATIO)
    parser.add_argument("--write-long-csv", type=int, choices=[0, 1], default=0)
    return parser


def validate_args(args: argparse.Namespace) -> Dict[str, pd.Timestamp]:
    dates = {
        "data_start": pd.Timestamp(args.data_start),
        "data_end": pd.Timestamp(args.data_end),
        "train_start": pd.Timestamp(args.train_start),
        "train_end": pd.Timestamp(args.train_end),
        "test_start": pd.Timestamp(args.test_start),
        "test_end": pd.Timestamp(args.test_end),
    }
    for start_name, end_name in [
        ("data_start", "data_end"),
        ("train_start", "train_end"),
        ("test_start", "test_end"),
    ]:
        if dates[end_name] < dates[start_name]:
            raise ValueError(f"{end_name} is earlier than {start_name}")
    if dates["train_start"] < dates["data_start"] or dates["train_end"] > dates["data_end"]:
        raise ValueError("Training period must be inside the data period")
    if dates["test_start"] < dates["data_start"] or dates["test_end"] > dates["data_end"]:
        raise ValueError("Test period must be inside the data period")
    if dates["train_end"] >= dates["test_start"]:
        raise ValueError("Training and test periods must not overlap and train must precede test")
    if args.lookback < 1:
        raise ValueError("--lookback must be >= 1")
    if not 0.0 <= args.valid_ratio_threshold <= 1.0:
        raise ValueError("--valid-ratio-threshold must be within [0, 1]")
    if args.min_train_samples < 0 or args.min_test_samples < 0:
        raise ValueError("Minimum train/test sample counts must be >= 0")
    if not 0.0 <= args.max_negative_qsim_ratio <= 1.0:
        raise ValueError("--max-negative-qsim-ratio must be within [0, 1]")
    return dates


def run_build(args: argparse.Namespace, logger: BuildLogger) -> List[Path]:
    parsed_dates = validate_args(args)
    cama_out_dir = Path(args.cama_out_dir)
    mapping_csv = Path(args.mapping_csv)
    grdc_dir = Path(args.grdc_dir)
    out_dir = Path(args.out_dir)
    series_dir = out_dir / "01_station_series"

    logger.log("[BUILD] reading and validating 5 arcmin mapping CSV")
    mapping_all = read_mapping_csv(mapping_csv)
    mapping_count = len(mapping_all)
    mapping = mapping_all
    logger.log(f"[BUILD] mapped stations: {mapping_count}")

    logger.log("[BUILD] loading annual Qsim cubes")
    dates, qsim_raw, qsim_info = extract_qsim_station_matrix(
        cama_out_dir,
        mapping,
        parsed_dates["data_start"],
        parsed_dates["data_end"],
        logger,
    )
    if not np.isfinite(qsim_raw).any():
        raise ValueError("Extracted qsim contains no finite values")

    finite_qsim = np.isfinite(qsim_raw)
    negative_mask = finite_qsim & (qsim_raw < 0.0)
    negative_count_total = int(negative_mask.sum())
    finite_count_total = int(finite_qsim.sum())
    negative_ratio_total = float(negative_count_total / finite_count_total)
    qsim_min_before_clip = float(np.min(qsim_raw[finite_qsim]))
    negative_count_by_station = negative_mask.sum(axis=0).astype(np.int64)
    finite_count_by_station = finite_qsim.sum(axis=0).astype(np.int64)
    negative_ratio_by_station = np.divide(
        negative_count_by_station,
        finite_count_by_station,
        out=np.zeros(len(mapping), dtype=np.float64),
        where=finite_count_by_station > 0,
    )
    min_before_clip_by_station = np.full(len(mapping), np.nan, dtype=np.float64)
    for station_index in range(len(mapping)):
        finite_station = qsim_raw[:, station_index][finite_qsim[:, station_index]]
        if finite_station.size:
            min_before_clip_by_station[station_index] = float(np.min(finite_station))
    logger.log(
        f"[QSIM] negative before clip: count={negative_count_total}, "
        f"ratio_over_finite={negative_ratio_total}, min={qsim_min_before_clip}"
    )

    # The US 5 arcmin source already stores discharge in m3/s. No *86400 conversion.
    qsim_all = np.where(finite_qsim, np.maximum(qsim_raw, 0.0), qsim_raw).astype(np.float32)

    logger.log("[BUILD] aligning GRDC Qobs without filling missing observations")
    qobs_all, qobs_covers_period = load_qobs_matrix(mapping, grdc_dir, dates, logger)
    qobs_valid_ratio = np.isfinite(qobs_all).sum(axis=0) / len(dates)
    qsim_valid_ratio = np.isfinite(qsim_all).sum(axis=0) / len(dates)
    lstm_valid = compute_lstm_valid_samples(qsim_all, qobs_all, args.lookback)
    train_mask_dates = np.asarray(
        (dates >= parsed_dates["train_start"]) & (dates <= parsed_dates["train_end"]),
        dtype=bool,
    )
    test_mask_dates = np.asarray(
        (dates >= parsed_dates["test_start"]) & (dates <= parsed_dates["test_end"]),
        dtype=bool,
    )
    if not train_mask_dates.any() or not test_mask_dates.any():
        raise ValueError("Training and test date masks must both contain at least one date")
    train_valid_samples = lstm_valid[train_mask_dates, :].sum(axis=0).astype(np.int64)
    test_valid_samples = lstm_valid[test_mask_dates, :].sum(axis=0).astype(np.int64)

    keep = (
        qobs_covers_period
        & (qobs_valid_ratio >= args.valid_ratio_threshold)
        & (train_valid_samples >= args.min_train_samples)
        & (test_valid_samples >= args.min_test_samples)
        & (negative_ratio_by_station <= args.max_negative_qsim_ratio)
    )
    if not keep.any():
        raise ValueError("No stations remain after observation/sample/negative-Qsim filtering")
    final = mapping.loc[keep].reset_index(drop=True)
    qsim = qsim_all[:, keep]
    qobs = qobs_all[:, keep]
    station_ids = station_ids_to_int64(final["station_id"].tolist())
    lat_cama = qsim_info["lat_cama"][keep]
    lon_cama = qsim_info["lon_cama"][keep]

    metadata = pd.DataFrame(
        {
            "station_id": station_ids,
            "lat": final["lat"].to_numpy(dtype=np.float64),
            "lon": final["lon"].to_numpy(dtype=np.float64),
            "lat_cama": lat_cama,
            "lon_cama": lon_cama,
            "x_index": final["x_index"].to_numpy(dtype=np.int64),
            "y_index": final["y_index"].to_numpy(dtype=np.int64),
            "qobs_valid_ratio_total": qobs_valid_ratio[keep],
            "qsim_valid_ratio_total": qsim_valid_ratio[keep],
            "train_valid_samples_for_lstm": train_valid_samples[keep],
            "test_valid_samples_for_lstm": test_valid_samples[keep],
            "qsim_negative_count_before_clip": negative_count_by_station[keep],
            "qsim_negative_ratio_before_clip": negative_ratio_by_station[keep],
            "qsim_min_before_clip": min_before_clip_by_station[keep],
            "qsim_mean_train": masked_column_mean(qsim, train_mask_dates),
            "qobs_mean_train": masked_column_mean(qobs, train_mask_dates),
            "qsim_mean_test": masked_column_mean(qsim, test_mask_dates),
            "qobs_mean_test": masked_column_mean(qobs, test_mask_dates),
        }
    )

    npz_path = series_dir / "station_qsim_qobs_2004_2010.npz"
    metadata_path = series_dir / "station_metadata_2004_2010.csv"
    build_summary_path = series_dir / "station_qsim_qobs_build_summary.txt"
    metrics_path = series_dir / "baseline_jiang_metrics_2004_2010.csv"
    baseline_summary_path = series_dir / "baseline_jiang_summary_2004_2010.txt"
    long_csv_path = series_dir / "station_daily_qsim_qobs_2004_2010.csv.gz"

    logger.log(f"[BUILD] retained stations: {len(final)}/{len(mapping)}")
    logger.log("[BUILD] writing NPZ and metadata")
    np.savez_compressed(
        npz_path,
        dates=np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"),
        station_ids=station_ids,
        lat=metadata["lat"].to_numpy(dtype=np.float32),
        lon=metadata["lon"].to_numpy(dtype=np.float32),
        lat_cama=metadata["lat_cama"].to_numpy(dtype=np.float32),
        lon_cama=metadata["lon_cama"].to_numpy(dtype=np.float32),
        x_index=metadata["x_index"].to_numpy(dtype=np.int64),
        y_index=metadata["y_index"].to_numpy(dtype=np.int64),
        qsim=qsim.astype(np.float32),
        qobs=qobs.astype(np.float32),
        qsim_scale=np.asarray(QSIM_SCALE, dtype=np.float32),
        qsim_source_units=np.asarray(QSIM_SOURCE_UNITS),
        qobs_units=np.asarray(QOBS_UNITS),
        time_shift_days=np.asarray(TIME_SHIFT_DAYS, dtype=np.int8),
        train_mask_dates=train_mask_dates,
        test_mask_dates=test_mask_dates,
        qsim_negative_count_before_clip_total=np.asarray(negative_count_total, dtype=np.int64),
        qsim_negative_ratio_before_clip_total=np.asarray(negative_ratio_total, dtype=np.float32),
        qsim_min_before_clip=np.asarray(qsim_min_before_clip, dtype=np.float32),
        max_negative_qsim_ratio=np.asarray(args.max_negative_qsim_ratio, dtype=np.float32),
    )
    metadata.to_csv(metadata_path, index=False)

    if args.write_long_csv:
        long_frame = pd.DataFrame(
            {
                "date": np.repeat(dates.strftime("%Y-%m-%d").to_numpy(), len(station_ids)),
                "station_id": np.tile(station_ids, len(dates)),
                "qsim": qsim.reshape(-1),
                "qobs": qobs.reshape(-1),
            }
        )
        long_frame.to_csv(long_csv_path, index=False, compression="gzip")

    logger.log("[BASELINE] reopening NPZ and computing Jiang CC/BR/RV KGE")
    metrics, baseline_summary = compute_baseline_from_npz(
        npz_path,
        parsed_dates["data_start"],
        parsed_dates["data_end"],
    )
    metrics.to_csv(metrics_path, index=False)
    baseline_lines = [
        "ERA5-Land US 5 arcmin CaMa baseline Jiang metrics summary",
        "",
        f"period: {args.data_start} .. {args.data_end}",
        "formula: KGE = 1 - sqrt((CC-1)^2 + (BR-1)^2 + (RV-1)^2)",
        "BR = mean(qsim) / mean(qobs)",
        "RV = (std(qsim)/mean(qsim)) / (std(qobs)/mean(qobs)); std uses ddof=0",
        "metrics use pairwise finite qsim/qobs values from the written NPZ",
        "",
        f"valid station count: {int(baseline_summary['valid_station_count'])}",
        f"station KGE median: {baseline_summary['median_KGE']}",
        f"station KGE mean: {baseline_summary['mean_KGE']}",
        f"KGE > 0 ratio: {baseline_summary['KGE_gt0_ratio']}",
        f"KGE > 0.5 ratio: {baseline_summary['KGE_gt05_ratio']}",
        f"CC median: {baseline_summary['median_CC']}",
        f"BR median: {baseline_summary['median_BR']}",
        f"RV median: {baseline_summary['median_RV']}",
        "",
    ]
    baseline_summary_path.write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    qsim_stats = finite_stats(qsim)
    qobs_stats = finite_stats(qobs)
    build_lines = [
        "ERA5-Land US 5 arcmin station Qsim/Qobs build summary",
        "",
        f"CAMA_OUT_DIR: {cama_out_dir}",
        f"MAPPING_CSV: {mapping_csv}",
        f"GRDC_DIR: {grdc_dir}",
        f"OUT_DIR: {out_dir}",
        "station source: mapping CSV only; evaluate/stn_list.txt is not used",
        "station extraction: direct zero-based x_index/y_index; no nearest-coordinate remapping",
        f"coordinate check: abs(NetCDF coordinate - CSV grid coordinate) < {COORD_TOLERANCE}",
        f"qsim extraction method: {qsim_info['extraction_method']}",
        "",
        f"data period: {args.data_start} .. {args.data_end}",
        f"date count: {len(dates)}",
        f"train period: {args.train_start} .. {args.train_end}",
        f"test period: {args.test_start} .. {args.test_end}",
        f"lookback: {args.lookback}",
        f"valid ratio threshold: {args.valid_ratio_threshold}",
        f"minimum train samples: {args.min_train_samples}",
        f"minimum test samples: {args.min_test_samples}",
        f"max negative qsim ratio: {args.max_negative_qsim_ratio}",
        "",
        f"qsim_scale: {QSIM_SCALE}",
        f"qsim_source_units: {QSIM_SOURCE_UNITS}",
        f"qobs_units: {QOBS_UNITS}",
        f"time_shift_days: {TIME_SHIFT_DAYS}",
        "qsim conversion: qsim = raw_outflw (no multiplication by 86400)",
        "finite negative qsim values are clipped to zero only in output/training arrays",
        "qsim negative ratios use finite qsim values as the denominator",
        "GRDC discharge <= -900 is NaN and is never filled with zero",
        "",
        f"mapping stations: {mapping_count}",
        f"processed stations: {len(mapping)}",
        f"stations covering full data period: {int(qobs_covers_period.sum())}",
        f"stations passing qobs valid ratio: {int((qobs_valid_ratio >= args.valid_ratio_threshold).sum())}",
        f"stations passing train sample minimum: {int((train_valid_samples >= args.min_train_samples).sum())}",
        f"stations passing test sample minimum: {int((test_valid_samples >= args.min_test_samples).sum())}",
        f"stations passing negative qsim ratio: {int((negative_ratio_by_station <= args.max_negative_qsim_ratio).sum())}",
        f"final retained stations: {len(final)}",
        f"qsim shape [time, station]: {qsim.shape}",
        f"qobs shape [time, station]: {qobs.shape}",
        f"qsim min/max/mean after clip: {qsim_stats['min']} / {qsim_stats['max']} / {qsim_stats['mean']}",
        f"qobs min/max/mean: {qobs_stats['min']} / {qobs_stats['max']} / {qobs_stats['mean']}",
        f"qsim negative count before clip total: {negative_count_total}",
        f"qsim negative ratio before clip total: {negative_ratio_total}",
        f"qsim min before clip: {qsim_min_before_clip}",
        "",
        "Annual CaMa files:",
    ]
    for item in qsim_info["yearly_info"]:
        build_lines.append(
            f"  {item['year']}: file={item['path']}; units={item['units']}; "
            f"original={item['original_start']}..{item['original_end']}; "
            f"shifted={item['corrected_start']}..{item['corrected_end']}; "
            f"raw_shape={item['raw_shape']}; station_shape={item['station_shape']}"
        )
    build_lines.extend(
        [
            "",
            f"baseline valid stations: {int(baseline_summary['valid_station_count'])}",
            f"baseline median KGE: {baseline_summary['median_KGE']}",
            f"baseline KGE > 0 ratio: {baseline_summary['KGE_gt0_ratio']}",
            f"baseline KGE > 0.5 ratio: {baseline_summary['KGE_gt05_ratio']}",
            "",
            f"NPZ: {npz_path}",
            f"metadata CSV: {metadata_path}",
            f"build summary: {build_summary_path}",
            f"baseline metrics CSV: {metrics_path}",
            f"baseline summary: {baseline_summary_path}",
            f"build log: {logger.path}",
        ]
    )
    if args.write_long_csv:
        build_lines.append(f"long CSV: {long_csv_path}")
    build_summary_path.write_text("\n".join(build_lines) + "\n", encoding="utf-8")

    logger.log(f"[BASELINE] valid stations: {int(baseline_summary['valid_station_count'])}")
    logger.log(f"[BASELINE] median KGE: {baseline_summary['median_KGE']}")
    logger.log(f"[BASELINE] KGE>0 ratio: {baseline_summary['KGE_gt0_ratio']}")
    logger.log(f"[BASELINE] KGE>0.5 ratio: {baseline_summary['KGE_gt05_ratio']}")

    output_paths = [npz_path, metadata_path, build_summary_path, metrics_path, baseline_summary_path]
    if args.write_long_csv:
        output_paths.append(long_csv_path)
    return output_paths


def main() -> None:
    args = build_parser().parse_args()
    series_dir = Path(args.out_dir) / "01_station_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    logger = BuildLogger(series_dir / "station_qsim_qobs_build.log")
    try:
        output_paths = run_build(args, logger)
    except Exception as exc:
        logger.log(f"[FATAL] {type(exc).__name__}: {exc}")
        raise

    logger.log("[BUILD] output files:")
    for path in output_paths:
        logger.log(f"  {path}")


if __name__ == "__main__":
    main()
