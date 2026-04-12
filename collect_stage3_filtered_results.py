#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_one(exp_dir: Path) -> dict | None:
    long_csv = exp_dir / "long_term_metrics.csv"
    short_csv = exp_dir / "short_term_metrics.csv"
    meta_json = exp_dir / "evaluation_metadata.json"

    if not long_csv.exists() or not short_csv.exists():
        return None

    long_df = pd.read_csv(long_csv)
    short_df = pd.read_csv(short_csv)
    if long_df.empty or short_df.empty:
        return None

    long_row = long_df.iloc[0].to_dict()
    short_row = short_df.iloc[0].to_dict()

    row = {
        "experiment": exp_dir.name,
        "short_rmse_avg": float(short_row.get("short_rmse_avg", float("nan"))),
        "long_rmse_avg": float(long_row.get("long_rmse_avg", float("nan"))),
        "late_horizon_rmse": float(long_row.get("late_horizon_rmse", float("nan"))),
        "policy_subset_late_rmse": float(long_row.get("policy_subset_late_rmse", float("nan"))),
        "trajectory_cum_mae": float(long_row.get("trajectory_cum_mae", float("nan"))),
        "peak_timing_mae_days": float(long_row.get("peak_timing_mae_days", float("nan"))),
    }

    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            cf = meta.get("country_filter", {})
            row["test_windows"] = int(cf.get("test_windows_after", meta.get("policy_subset_n", 0) * 4))
            row["countries_dropped_n"] = int(cf.get("countries_dropped_n", 0))
        except Exception:
            row["test_windows"] = float("nan")
            row["countries_dropped_n"] = float("nan")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect stage3 filtered experiment metrics")
    ap.add_argument("--results_root", type=str, default="results/stage3_extended_filtered")
    ap.add_argument("--out_csv", type=str, default="results/stage3_extended_filtered/stage3_filtered_summary.csv")
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for p in sorted(root.iterdir() if root.exists() else []):
        if not p.is_dir():
            continue
        r = load_one(p)
        if r is not None:
            rows.append(r)

    if not rows:
        raise SystemExit(f"No experiment results found under {root}")

    df = pd.DataFrame(rows).sort_values(["late_horizon_rmse", "long_rmse_avg", "short_rmse_avg"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("=" * 76)
    print("STAGE3 FILTERED SUMMARY")
    print("=" * 76)
    print(df.to_string(index=False))
    print("=" * 76)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
