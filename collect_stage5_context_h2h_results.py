#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _read_row(df: pd.DataFrame, model: str) -> Optional[dict]:
    row = df[df["model"] == model]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def load_one(exp_dir: Path) -> Optional[dict]:
    all_csv = exp_dir / "all_metrics_full.csv"
    short_csv = exp_dir / "short_term_metrics.csv"
    long_csv = exp_dir / "long_term_metrics.csv"
    pvsp_csv = exp_dir / "planning_vs_persistence.csv"
    if not all_csv.exists() or not short_csv.exists() or not long_csv.exists():
        return None

    all_df = pd.read_csv(all_csv)
    short_df = pd.read_csv(short_csv)
    long_df = pd.read_csv(long_csv)
    if all_df.empty or short_df.empty or long_df.empty:
        return None

    all_row = _read_row(all_df, "crt")
    short_row = _read_row(short_df, "crt")
    long_row = _read_row(long_df, "crt")
    if all_row is None or short_row is None or long_row is None:
        return None

    row: Dict[str, object] = {
        "experiment": exp_dir.name,
        "overall_rmse": float(all_row.get("overall_rmse", float("nan"))),
        "overall_mae": float(all_row.get("overall_mae", float("nan"))),
        "short_rmse_avg": float(short_row.get("short_rmse_avg", float("nan"))),
        "long_rmse_avg": float(long_row.get("long_rmse_avg", float("nan"))),
        "late_horizon_rmse": float(long_row.get("late_horizon_rmse", float("nan"))),
        "policy_subset_late_rmse": float(long_row.get("policy_subset_late_rmse", float("nan"))),
        "trajectory_cum_mae": float(long_row.get("trajectory_cum_mae", float("nan"))),
        "peak_timing_mae_days": float(long_row.get("peak_timing_mae_days", float("nan"))),
        "peak_magnitude_mae": float(long_row.get("peak_magnitude_mae", float("nan"))),
    }

    if pvsp_csv.exists():
        pvsp_df = pd.read_csv(pvsp_csv)
        pvsp_row = _read_row(pvsp_df, "crt")
        if pvsp_row is not None:
            row["delta_late_rmse_vs_persistence"] = float(
                pvsp_row.get("delta_late_rmse_vs_persistence", float("nan"))
            )
            row["late_win_rate_vs_persistence"] = float(
                pvsp_row.get("late_win_rate_vs_persistence", float("nan"))
            )
            row["delta_policy_subset_late_rmse_vs_persistence"] = float(
                pvsp_row.get("delta_policy_subset_late_rmse_vs_persistence", float("nan"))
            )
            row["policy_subset_late_win_rate_vs_persistence"] = float(
                pvsp_row.get("policy_subset_late_win_rate_vs_persistence", float("nan"))
            )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Stage5 context h2h-with-baselines metrics.")
    ap.add_argument("--results_root", type=str, default="results/stage5_context_h2h_with_baselines")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/stage5_context_h2h_with_baselines/stage5_context_h2h_summary.csv",
    )
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
        raise SystemExit(f"No Stage5 h2h results found under {root}")

    df = pd.DataFrame(rows).sort_values(
        ["late_horizon_rmse", "long_rmse_avg", "short_rmse_avg", "overall_rmse"]
    ).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("=" * 100)
    print("STAGE5 CONTEXT H2H SUMMARY (CRT row per experiment)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
