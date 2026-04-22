#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _copy_if_exists(src: Path, dst: Path, copied: List[str], missing: List[str]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    else:
        missing.append(str(src))


def _copy_glob(
    root: Path,
    patterns: Iterable[str],
    out_dir: Path,
    copied: List[str],
    missing: List[str],
) -> None:
    any_found = False
    for pat in patterns:
        matches = list(root.glob(pat))
        if matches:
            any_found = True
            for src in matches:
                dst = out_dir / src.name
                _copy_if_exists(src, dst, copied, missing)
    if not any_found:
        for pat in patterns:
            missing.append(str(root / pat))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a clean Stage5 context report pack for Overleaf.")
    ap.add_argument("--h2h_root", type=str, default="results/stage5_context_h2h_with_baselines")
    ap.add_argument(
        "--summary_csv",
        type=str,
        default="results/stage5_context_h2h_with_baselines/stage5_context_h2h_summary.csv",
    )
    ap.add_argument("--best_experiment", type=str, default="")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/overleaf_clean_pack_stage5_context",
    )
    args = ap.parse_args()

    h2h_root = Path(args.h2h_root)
    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_fig = out_dir / "figures"
    out_tab = out_dir / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    if args.best_experiment.strip():
        best_exp = args.best_experiment.strip()
    else:
        if not summary_csv.exists():
            raise FileNotFoundError(
                f"Summary CSV not found: {summary_csv}. Run collect_stage5_context_h2h_results.py first."
            )
        s = pd.read_csv(summary_csv)
        if s.empty:
            raise ValueError(f"Summary CSV is empty: {summary_csv}")
        best_exp = str(s.sort_values("late_horizon_rmse").iloc[0]["experiment"])

    src_dir = h2h_root / best_exp
    if not src_dir.exists():
        raise FileNotFoundError(f"Best experiment directory not found: {src_dir}")

    copied: List[str] = []
    missing: List[str] = []

    # Core tables
    for name in [
        "table_main_results.csv",
        "table_main_results.md",
        "table_policy_change_subset.csv",
        "table_policy_change_subset.md",
        "table_regime_breakdown.csv",
        "table_regime_breakdown.md",
        "short_term_metrics.csv",
        "long_term_metrics.csv",
        "planning_vs_persistence.csv",
        "shape_metrics.csv",
        "stability_metrics.csv",
        "utility_metrics.csv",
        "incidence_regime_metrics.csv",
        "all_metrics_full.csv",
        "evaluation_metadata.json",
    ]:
        _copy_if_exists(src_dir / name, out_tab / name, copied, missing)

    # Cross-experiment summaries
    _copy_if_exists(summary_csv, out_tab / summary_csv.name, copied, missing)

    sweep_csv = Path("results/stage5_context_extended/stage5_context_summary.csv")
    _copy_if_exists(sweep_csv, out_tab / sweep_csv.name, copied, missing)

    # Core figures
    _copy_if_exists(src_dir / "fig_horizon_profile.png", out_fig / "fig_horizon_profile.png", copied, missing)

    _copy_glob(
        src_dir,
        patterns=[
            "trajectory_focus_beats_reference_*.png",
            "trajectory_focus_loses_to_reference_*.png",
            "fig_regime_late_rmse_bars.png",
            "fig_peak_timing_mae.png",
        ],
        out_dir=out_fig,
        copied=copied,
        missing=missing,
    )

    readme_lines = [
        "# Stage5 Context Report Pack",
        "",
        f"- Selected best experiment: `{best_exp}`",
        f"- Source directory: `{src_dir}`",
        "",
        "## Copied files",
    ]
    readme_lines.extend([f"- {p}" for p in copied])
    readme_lines.extend(["", "## Missing files"])
    if missing:
        readme_lines.extend([f"- {p}" for p in missing])
    else:
        readme_lines.append("- none")
    readme_lines.append("")

    readme = out_dir / "README_STAGE5_REPORT_PACK.md"
    readme.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print("=" * 90)
    print("STAGE5 CONTEXT REPORT PACK")
    print("=" * 90)
    print(f"Best experiment: {best_exp}")
    print(f"Output directory: {out_dir}")
    print(f"Copied files: {len(copied)}")
    print(f"Missing files: {len(missing)}")
    print(f"README: {readme}")
    print("=" * 90)


if __name__ == "__main__":
    main()
