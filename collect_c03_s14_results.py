#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _rmse_cols(df: pd.DataFrame, start: int, end: int) -> List[str]:
    cols = []
    for h in range(start, end + 1):
        c = f"rmse_h{h}"
        if c in df.columns:
            cols.append(c)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize h14 head-to-head metrics (short vs mid).")
    ap.add_argument(
        "--metrics_csv",
        type=str,
        default="results/stage4_h35_week46_h14/c03_s14_vs_c03_h2h/all_metrics_full.csv",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/stage4_h35_week46_h14/c03_s14_vs_c03_h2h/summary_short_mid_h14.csv",
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="results/stage4_h35_week46_h14/c03_s14_vs_c03_h2h/summary_short_mid_h14.md",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    req = ["model", "overall_rmse"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {args.metrics_csv}")

    short_cols = _rmse_cols(df, 1, 7)
    mid_cols = _rmse_cols(df, 8, 14)
    if len(short_cols) < 7 or len(mid_cols) < 7:
        raise ValueError(
            f"Expected rmse_h1..rmse_h14 in metrics. Found short={len(short_cols)} cols, mid={len(mid_cols)} cols."
        )

    out = df[["model", "overall_rmse"]].copy()
    out["short_rmse_1_7"] = df[short_cols].mean(axis=1)
    out["mid_rmse_8_14"] = df[mid_cols].mean(axis=1)

    if "persistence" in set(out["model"]):
        p_short = float(out.loc[out["model"] == "persistence", "short_rmse_1_7"].iloc[0])
        p_mid = float(out.loc[out["model"] == "persistence", "mid_rmse_8_14"].iloc[0])
        out["delta_short_vs_persistence"] = out["short_rmse_1_7"] - p_short
        out["delta_mid_vs_persistence"] = out["mid_rmse_8_14"] - p_mid

    out = out.sort_values("mid_rmse_8_14").reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    disp = out.copy()
    for c in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].map(lambda x: f"{float(x):.4f}")
        else:
            disp[c] = disp[c].astype(str)
    header = "| " + " | ".join(disp.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(disp.columns)) + " |"
    rows = ["| " + " | ".join(r.tolist()) + " |" for _, r in disp.iterrows()]
    out_md.write_text("\n".join([header, sep] + rows) + "\n", encoding="utf-8")

    print("Saved:")
    print(f"  - {out_csv}")
    print(f"  - {out_md}")


if __name__ == "__main__":
    main()
