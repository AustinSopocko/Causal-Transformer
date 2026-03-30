#!/usr/bin/env python
"""
Collect Stage 2 Oxford benchmark outputs into a single summary table.

Usage:
  python collect_stage2_results.py --results_root results/stage2
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def extract_model_rmse(df: pd.DataFrame, model_name: str) -> Optional[float]:
    row = df[df["model"] == model_name]
    if row.empty:
        return None
    return float(row["overall_rmse"].iloc[0])


def extract_model_metric(df: pd.DataFrame, model_name: str, metric_col: str) -> Optional[float]:
    row = df[df["model"] == model_name]
    if row.empty or metric_col not in df.columns:
        return None
    value = pd.to_numeric(row[metric_col], errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def collect_results(results_root: Path, extended_root: Optional[Path] = None) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for exp_dir in sorted(results_root.glob("*")):
        if not exp_dir.is_dir():
            continue
        csv_path = exp_dir / "oxford_benchmarks.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        crt_rmse = extract_model_rmse(df, "CRT")
        mean_rmse = extract_model_rmse(df, "mean")
        persistence_rmse = extract_model_rmse(df, "persistence")
        if crt_rmse is None:
            continue
        row: Dict[str, float | str] = {
            "experiment": exp_dir.name,
            "crt_rmse": crt_rmse,
            "persistence_rmse": persistence_rmse if persistence_rmse is not None else float("nan"),
            "mean_rmse": mean_rmse if mean_rmse is not None else float("nan"),
            "gap_to_persistence": (
                crt_rmse - persistence_rmse if persistence_rmse is not None else float("nan")
            ),
            "improvement_vs_mean_pct": (
                100.0 * (mean_rmse - crt_rmse) / mean_rmse if mean_rmse is not None and mean_rmse > 0 else float("nan")
            ),
        }

        if extended_root is not None:
            ext_csv = extended_root / exp_dir.name / "all_metrics_full.csv"
            if ext_csv.exists():
                ext_df = pd.read_csv(ext_csv)
                model_label = "crt" if (ext_df["model"] == "crt").any() else "CRT"
                crt_long = extract_model_metric(ext_df, model_label, "long_rmse_avg")
                crt_late = extract_model_metric(ext_df, model_label, "late_horizon_rmse")
                crt_policy_late = extract_model_metric(ext_df, model_label, "policy_subset_late_rmse")

                persist_long = extract_model_metric(ext_df, "persistence", "long_rmse_avg")
                persist_late = extract_model_metric(ext_df, "persistence", "late_horizon_rmse")
                persist_policy_late = extract_model_metric(ext_df, "persistence", "policy_subset_late_rmse")

                row["crt_long_rmse"] = crt_long if crt_long is not None else float("nan")
                row["persistence_long_rmse"] = persist_long if persist_long is not None else float("nan")
                row["long_gap_to_persistence"] = (
                    crt_long - persist_long if crt_long is not None and persist_long is not None else float("nan")
                )
                row["crt_late_rmse"] = crt_late if crt_late is not None else float("nan")
                row["persistence_late_rmse"] = persist_late if persist_late is not None else float("nan")
                row["late_gap_to_persistence"] = (
                    crt_late - persist_late if crt_late is not None and persist_late is not None else float("nan")
                )
                row["crt_policy_subset_late_rmse"] = (
                    crt_policy_late if crt_policy_late is not None else float("nan")
                )
                row["persistence_policy_subset_late_rmse"] = (
                    persist_policy_late if persist_policy_late is not None else float("nan")
                )
                row["policy_subset_late_gap_to_persistence"] = (
                    crt_policy_late - persist_policy_late
                    if crt_policy_late is not None and persist_policy_late is not None
                    else float("nan")
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        if "late_gap_to_persistence" in out.columns and out["late_gap_to_persistence"].notna().any():
            out = out.sort_values(
                ["late_gap_to_persistence", "long_gap_to_persistence", "gap_to_persistence"],
                ascending=[True, True, True],
            ).reset_index(drop=True)
        else:
            out = out.sort_values(["crt_rmse", "gap_to_persistence"], ascending=[True, True]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Stage 2 Oxford benchmark CSV results")
    parser.add_argument("--results_root", type=str, default="results/stage2")
    parser.add_argument("--extended_root", type=str, default="results/stage2_extended")
    parser.add_argument("--out_csv", type=str, default="results/stage2/stage2_summary.csv")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    extended_root = Path(args.extended_root) if args.extended_root else None
    summary = collect_results(results_root, extended_root=extended_root)
    if summary.empty:
        print(f"No stage2 benchmark results found under {results_root}")
        return

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print("=" * 72)
    print("STAGE 2 SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False, justify="left"))
    print("=" * 72)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
