#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


BASELINE_NAMES = {
    "mean",
    "persistence",
    "seasonal_naive_7d",
    "linear_trend",
    "last_7day_mean",
    "ar_ridge_lag",
    "arx_ridge_lag_policy",
    "boosted_lag_policy",
    "lstm_seq2seq",
}


def _extract_seed(exp_name: str) -> Optional[int]:
    digits = "".join(ch for ch in exp_name if ch.isdigit())
    return int(digits) if digits else None


def load_one(exp_dir: Path) -> Optional[Dict[str, float | str | int]]:
    long_csv = exp_dir / "long_term_metrics.csv"
    short_csv = exp_dir / "short_term_metrics.csv"
    pvsp_csv = exp_dir / "planning_vs_persistence.csv"
    if not long_csv.exists() or not short_csv.exists() or not pvsp_csv.exists():
        return None

    long_df = pd.read_csv(long_csv)
    short_df = pd.read_csv(short_csv)
    pvsp_df = pd.read_csv(pvsp_csv)
    if long_df.empty or short_df.empty or pvsp_df.empty:
        return None

    exp_name = exp_dir.name
    short_row = short_df[short_df["model"] == exp_name]
    if short_row.empty:
        short_row = short_df[~short_df["model"].isin(BASELINE_NAMES)]
        if short_row.empty:
            return None
        short_row = short_row.iloc[[0]]
    long_row = long_df[long_df["model"] == short_row.iloc[0]["model"]]
    if long_row.empty:
        return None
    long_row = long_row.iloc[0]

    pvsp_row = pvsp_df[pvsp_df["model"] == short_row.iloc[0]["model"]]
    if pvsp_row.empty:
        return None
    pvsp_row = pvsp_row.iloc[0]

    return {
        "experiment": exp_name,
        "seed": _extract_seed(exp_name),
        "model": str(short_row.iloc[0]["model"]),
        "short_rmse_avg": float(short_row.iloc[0].get("short_rmse_avg", float("nan"))),
        "long_rmse_avg": float(long_row.get("long_rmse_avg", float("nan"))),
        "late_horizon_rmse": float(long_row.get("late_horizon_rmse", float("nan"))),
        "policy_subset_late_rmse": float(long_row.get("policy_subset_late_rmse", float("nan"))),
        "delta_late_rmse_vs_persistence": float(pvsp_row.get("delta_late_rmse_vs_persistence", float("nan"))),
        "late_win_rate_vs_persistence": float(pvsp_row.get("late_win_rate_vs_persistence", float("nan"))),
        "delta_policy_subset_late_rmse_vs_persistence": float(
            pvsp_row.get("delta_policy_subset_late_rmse_vs_persistence", float("nan"))
        ),
        "policy_subset_late_win_rate_vs_persistence": float(
            pvsp_row.get("policy_subset_late_win_rate_vs_persistence", float("nan"))
        ),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "short_rmse_avg",
        "long_rmse_avg",
        "late_horizon_rmse",
        "policy_subset_late_rmse",
        "delta_late_rmse_vs_persistence",
        "late_win_rate_vs_persistence",
        "delta_policy_subset_late_rmse_vs_persistence",
        "policy_subset_late_win_rate_vs_persistence",
    ]
    rows: List[Dict[str, float | str | int]] = []
    for m in metric_cols:
        s = pd.to_numeric(df[m], errors="coerce")
        rows.append(
            {
                "metric": m,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "n": int(s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Stage4 c03 seed-stability results.")
    ap.add_argument("--results_root", type=str, default="results/stage4_seed_stability")
    ap.add_argument(
        "--out_per_seed_csv",
        type=str,
        default="results/stage4_seed_stability/seed_stability_per_seed.csv",
    )
    ap.add_argument(
        "--out_summary_csv",
        type=str,
        default="results/stage4_seed_stability/seed_stability_summary.csv",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    rows: List[Dict[str, float | str | int]] = []
    for p in sorted(root.iterdir() if root.exists() else []):
        if not p.is_dir():
            continue
        r = load_one(p)
        if r is not None:
            rows.append(r)

    if not rows:
        raise SystemExit(f"No seed-run results found under {root}")

    per_seed = pd.DataFrame(rows).sort_values(["seed", "experiment"]).reset_index(drop=True)
    summary = build_summary(per_seed)

    out_per_seed = Path(args.out_per_seed_csv)
    out_summary = Path(args.out_summary_csv)
    out_per_seed.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_per_seed, index=False)
    summary.to_csv(out_summary, index=False)

    print("=" * 90)
    print("STAGE4 SEED STABILITY (PER-SEED)")
    print("=" * 90)
    print(per_seed.to_string(index=False))
    print("=" * 90)
    print("STAGE4 SEED STABILITY (SUMMARY)")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("=" * 90)
    print(f"Saved {out_per_seed}")
    print(f"Saved {out_summary}")


if __name__ == "__main__":
    main()
