#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _safe_float(row: Dict, key: str) -> float:
    return float(row.get(key, float("nan")))


def _parse_config_hints(name: str) -> Dict[str, object]:
    return {
        "config_name": name,
        "uses_country_id_embedding": ("id_only" in name) or ("plus_id" in name),
        "uses_explicit_country_covariates": ("context_only" in name) or ("plus_id" in name),
    }


def load_one(exp_dir: Path) -> Optional[dict]:
    all_csv = exp_dir / "all_metrics_full.csv"
    long_csv = exp_dir / "long_term_metrics.csv"
    short_csv = exp_dir / "short_term_metrics.csv"
    meta_json = exp_dir / "evaluation_metadata.json"
    if not all_csv.exists() or not long_csv.exists() or not short_csv.exists():
        return None

    all_df = pd.read_csv(all_csv)
    long_df = pd.read_csv(long_csv)
    short_df = pd.read_csv(short_csv)
    if all_df.empty or long_df.empty or short_df.empty:
        return None

    all_row = all_df.iloc[0].to_dict()
    long_row = long_df.iloc[0].to_dict()
    short_row = short_df.iloc[0].to_dict()

    row: Dict[str, object] = {
        "experiment": exp_dir.name,
        "overall_rmse": _safe_float(all_row, "overall_rmse"),
        "overall_mae": _safe_float(all_row, "overall_mae"),
        "short_rmse_avg": _safe_float(short_row, "short_rmse_avg"),
        "long_rmse_avg": _safe_float(long_row, "long_rmse_avg"),
        "late_horizon_rmse": _safe_float(long_row, "late_horizon_rmse"),
        "policy_subset_late_rmse": _safe_float(long_row, "policy_subset_late_rmse"),
        "trajectory_cum_mae": _safe_float(long_row, "trajectory_cum_mae"),
        "peak_timing_mae_days": _safe_float(long_row, "peak_timing_mae_days"),
        "peak_magnitude_mae": _safe_float(long_row, "peak_magnitude_mae"),
    }
    row.update(_parse_config_hints(exp_dir.name))

    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            country_filter = meta.get("country_filter", {})
            row["test_windows"] = int(country_filter.get("test_windows_after", 0))
            row["countries_dropped_n"] = int(country_filter.get("countries_dropped_n", 0))
            row["policy_subset_n"] = int(meta.get("policy_subset_n", 0))
            row["policy_subset_ratio"] = float(meta.get("policy_subset_ratio", 0.0))

        except Exception:
            pass

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Stage5 country-context sweep metrics.")
    ap.add_argument("--results_root", type=str, default="results/stage5_context_extended")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/stage5_context_extended/stage5_context_summary.csv",
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
        raise SystemExit(f"No experiment results found under {root}")

    df = pd.DataFrame(rows).sort_values(
        ["late_horizon_rmse", "long_rmse_avg", "short_rmse_avg", "overall_rmse"]
    ).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("=" * 84)
    print("STAGE5 CONTEXT SWEEP SUMMARY")
    print("=" * 84)
    print(df.to_string(index=False))
    print("=" * 84)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
