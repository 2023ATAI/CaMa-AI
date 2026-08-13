#!/usr/bin/env python
"""Scan inputs for the CaMa-PyTorch + LSTM post-processing example."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr


STN_LIST = "evaluate/stn_list.txt"
GRDC_DIR = "data/GRDC/GRDC_US/GRDC_Day"
DATA_START = "2004-01-02"
DATA_END = "2010-12-31"
VALID_RATIO_THRESHOLD = 0.8

REQUIRED_STATION_COLUMNS = ["ID", "lat", "lon", "Flag", "lon_cama", "lat_cama"]


def parse_bool_flag(value: Any) -> bool:
    """Parse station Flag values that may arrive as bools, numbers, or strings."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))

    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse station Flag value as boolean: {value!r}")


def station_id_to_str(station_id: Any) -> str:
    if pd.isna(station_id):
        raise ValueError("Station ID is missing.")
    try:
        as_float = float(station_id)
    except (TypeError, ValueError):
        return str(station_id).strip()
    if as_float.is_integer():
        return str(int(as_float))
    return str(station_id).strip()


def read_station_list(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Station list not found: {path}")

    stations = pd.read_csv(path)
    missing = [col for col in REQUIRED_STATION_COLUMNS if col not in stations.columns]
    if missing:
        raise ValueError(
            f"Station list {path} is missing required columns: {', '.join(missing)}"
        )

    stations = stations.copy()
    stations["Flag_parsed"] = stations["Flag"].map(parse_bool_flag)
    return stations


def scan_grdc_station(
    station_id: Any,
    grdc_dir: Path,
    data_start: str,
    data_end: str,
    valid_ratio_threshold: float,
) -> Dict[str, Any]:
    station_id_text = station_id_to_str(station_id)
    path = grdc_dir / f"{station_id_text}_Q_Day.Cmd.nc"
    result = {
        "station_id": station_id_text,
        "file_exists": path.exists(),
        "time_start": "",
        "time_end": "",
        "covers_period": False,
        "valid_ratio": np.nan,
        "eligible_by_obs": False,
        "path": str(path),
        "error": "",
    }
    if not path.exists():
        return result

    expected_dates = pd.date_range(data_start, data_end, freq="D")
    try:
        with xr.open_dataset(path) as ds:
            if "time" not in ds.coords and "time" not in ds.dims:
                raise ValueError("missing time coordinate")
            if "discharge" not in ds:
                raise ValueError("missing discharge variable")

            time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
            if len(time_index) == 0:
                raise ValueError("empty time coordinate")

            result["time_start"] = time_index.min().strftime("%Y-%m-%d")
            result["time_end"] = time_index.max().strftime("%Y-%m-%d")
            result["covers_period"] = (
                time_index.min() <= pd.Timestamp(data_start)
                and time_index.max() >= pd.Timestamp(data_end)
            )

            discharge = ds["discharge"].squeeze(drop=True)
            if "time" not in discharge.dims:
                raise ValueError(
                    f"discharge has no time dimension after squeeze; dims={discharge.dims}"
                )

            discharge = discharge.where(discharge > -900)
            discharge = discharge.sel(time=slice(data_start, data_end))
            discharge = discharge.reindex(time=expected_dates)
            values = discharge.to_numpy()
            valid_ratio = float(np.isfinite(values).sum() / len(expected_dates))
            result["valid_ratio"] = valid_ratio
            result["eligible_by_obs"] = bool(
                result["covers_period"] and valid_ratio >= valid_ratio_threshold
            )
    except Exception as exc:  # noqa: BLE001 - report per-station scan failures.
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def find_yearly_outflw_files(cama_cache_dir: Path, years: List[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        matches = sorted(cama_cache_dir.glob(f"*outflw*{year}*.nc"))
        row = {
            "year": year,
            "found_count": len(matches),
            "status": "ok" if len(matches) == 1 else ("missing" if not matches else "multiple"),
            "path": str(matches[0]) if len(matches) == 1 else "",
            "matched_paths": ";".join(str(path) for path in matches),
            "has_outflw": False,
            "dims": "",
            "shape": "",
            "units": "",
            "time_start": "",
            "time_end": "",
            "error": "",
        }

        if len(matches) == 1:
            try:
                with xr.open_dataset(matches[0]) as ds:
                    if "outflw" not in ds:
                        raise ValueError("missing outflw variable")
                    outflw = ds["outflw"]
                    row["has_outflw"] = True
                    row["dims"] = ",".join(outflw.dims)
                    row["shape"] = ",".join(str(size) for size in outflw.shape)
                    row["units"] = str(outflw.attrs.get("units", ""))
                    if "time" in outflw.coords or "time" in outflw.dims:
                        time_index = pd.DatetimeIndex(pd.to_datetime(outflw["time"].values))
                        if len(time_index) > 0:
                            row["time_start"] = time_index.min().strftime("%Y-%m-%d")
                            row["time_end"] = time_index.max().strftime("%Y-%m-%d")
            except Exception as exc:  # noqa: BLE001 - keep scanning other years.
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan station, GRDC, and CaMa outflw inputs for LSTM preprocessing."
    )
    parser.add_argument(
        "--cama-out-dir",
        type=Path,
        required=True,
        help="Path to the CaMa-PyTorch simulation output directory.",
    )
    parser.add_argument("--stn-list", default=STN_LIST)
    parser.add_argument("--grdc-dir", default=GRDC_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for generated LSTM preprocessing outputs.",
    )
    parser.add_argument("--data-start", default=DATA_START)
    parser.add_argument("--data-end", default=DATA_END)
    parser.add_argument("--valid-ratio-threshold", type=float, default=VALID_RATIO_THRESHOLD)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cama_cache_dir = args.cama_out_dir
    stn_list = Path(args.stn_list)
    grdc_dir = Path(args.grdc_dir)
    out_dir = args.out_dir
    scan_dir = out_dir / "00_scan_inputs"
    scan_dir.mkdir(parents=True, exist_ok=True)

    data_start = pd.Timestamp(args.data_start)
    data_end = pd.Timestamp(args.data_end)
    if data_end < data_start:
        raise ValueError(f"data_end {args.data_end} is earlier than data_start {args.data_start}")

    stations = read_station_list(stn_list)
    flag_true = stations.loc[stations["Flag_parsed"]].copy()

    coverage_rows = [
        scan_grdc_station(
            row.ID,
            grdc_dir,
            args.data_start,
            args.data_end,
            args.valid_ratio_threshold,
        )
        for row in flag_true.itertuples(index=False)
    ]
    coverage = pd.DataFrame(coverage_rows)
    coverage_path = scan_dir / "grdc_station_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    years = list(range(data_start.year, data_end.year + 1))
    inventory = find_yearly_outflw_files(cama_cache_dir, years)
    inventory_path = scan_dir / "cama_outflw_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    eligible_ids = set(
        coverage.loc[
            coverage["file_exists"]
            & coverage["covers_period"]
            & (coverage["valid_ratio"] >= args.valid_ratio_threshold),
            "station_id",
        ].astype(str)
    )
    coverage_by_id = coverage.set_index(coverage["station_id"].astype(str), drop=False)
    eligible = flag_true.loc[
        flag_true["ID"].map(station_id_to_str).isin(eligible_ids)
    ].copy()
    eligible["valid_ratio"] = [
        coverage_by_id.loc[station_id_to_str(station_id), "valid_ratio"]
        for station_id in eligible["ID"]
    ]
    eligible["grdc_path"] = [
        coverage_by_id.loc[station_id_to_str(station_id), "path"]
        for station_id in eligible["ID"]
    ]

    keep_cols = [
        "ID",
        "lat",
        "lon",
        "lon_cama",
        "lat_cama",
        "obs_syear",
        "obs_eyear",
        "valid_ratio",
        "grdc_path",
    ]
    for col in keep_cols:
        if col not in eligible.columns:
            eligible[col] = np.nan
    eligible = eligible[keep_cols]
    eligible_path = scan_dir / "eligible_stations_initial.csv"
    eligible.to_csv(eligible_path, index=False)

    grdc_exists_count = int(coverage["file_exists"].sum()) if not coverage.empty else 0
    covers_count = int(coverage["covers_period"].sum()) if not coverage.empty else 0
    valid_count = int((coverage["valid_ratio"] >= args.valid_ratio_threshold).sum())

    summary_lines = [
        "CaMa-PyTorch + LSTM input scan summary",
        "",
        f"CAMA_CACHE_DIR: {cama_cache_dir}",
        f"STN_LIST: {stn_list}",
        f"GRDC_DIR: {grdc_dir}",
        f"OUT_DIR: {out_dir}",
        f"DATA_START: {args.data_start}",
        f"DATA_END: {args.data_end}",
        f"VALID_RATIO_THRESHOLD: {args.valid_ratio_threshold}",
        "",
        f"stn_list total stations: {len(stations)}",
        f"Flag=True stations: {len(flag_true)}",
        f"GRDC files found: {grdc_exists_count}",
        f"GRDC stations covering period: {covers_count}",
        f"GRDC stations valid_ratio >= threshold: {valid_count}",
        f"initial eligible stations: {len(eligible)}",
        "",
        "CaMa outflw inventory:",
    ]
    for row in inventory.itertuples(index=False):
        summary_lines.append(
            f"  {row.year}: status={row.status}, found_count={row.found_count}, "
            f"has_outflw={row.has_outflw}, time={row.time_start}..{row.time_end}, "
            f"path={row.path or row.matched_paths}"
        )
        if row.error:
            summary_lines.append(f"    error: {row.error}")
    summary_lines.extend(
        [
            "",
            f"input_scan_summary: {scan_dir / 'input_scan_summary.txt'}",
            f"grdc_station_coverage: {coverage_path}",
            f"cama_outflw_inventory: {inventory_path}",
            f"eligible_stations_initial: {eligible_path}",
        ]
    )

    summary_path = scan_dir / "input_scan_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"stn_list total stations: {len(stations)}")
    print(f"Flag=True stations: {len(flag_true)}")
    print(f"GRDC files found: {grdc_exists_count}")
    print(f"GRDC stations covering {args.data_start}-{args.data_end}: {covers_count}")
    print(f"valid_ratio >= {args.valid_ratio_threshold}: {valid_count}")
    print("CaMa outflw inventory:")
    for row in inventory.itertuples(index=False):
        print(
            f"  {row.year}: status={row.status}, found_count={row.found_count}, "
            f"time={row.time_start}..{row.time_end}, path={row.path or row.matched_paths}"
        )
    print("Output files:")
    print(f"  {summary_path}")
    print(f"  {coverage_path}")
    print(f"  {inventory_path}")
    print(f"  {eligible_path}")


if __name__ == "__main__":
    main()
