#!/usr/bin/env python
"""
Stage 1 sanity and fairness audit for Oxford CRT benchmarks.

Usage:
    python stage1_oxford.py \
      --checkpoint checkpoints/oxford/best_crt.pt \
      --oxford_csv data/oxford/oxford_panel.csv \
      --config src/configs/oxford_config.yaml \
      --output_dir results/stage1
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from crt.model import CRTModel
from crt.rollout import rollout
from run_rq1 import load_checkpoint
from src.data.normalise import (
    OutcomeScaler,
    apply_outcome_scaler,
    fit_outcome_scaler,
    inverse_transform_outcomes,
)
from src.data.oxford_loader import clean_oxford, load_oxford_csv, select_features
from src.data.panel_windows import PanelWindows, build_country_index, make_windows
from src.train.splits import time_split_window_indices


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, List[float] | float]:
    sq_err = (y_pred - y_true) ** 2
    overall = float(torch.sqrt(torch.mean(sq_err)).item())
    per_horizon = torch.sqrt(torch.mean(sq_err, dim=(0, 2))).cpu().tolist()
    return {"overall": overall, "per_horizon": [float(v) for v in per_horizon]}


def per_outcome_rmse(y_pred: torch.Tensor, y_true: torch.Tensor, outcome_cols: List[str]) -> pd.DataFrame:
    sq_err = (y_pred - y_true) ** 2  # (N, H, d_y)
    rows: List[Dict[str, float | str | int]] = []
    horizon = sq_err.shape[1]
    for h in range(horizon):
        row: Dict[str, float | str | int] = {"horizon": h + 1}
        for j, name in enumerate(outcome_cols):
            row[f"rmse_{name}"] = float(torch.sqrt(torch.mean(sq_err[:, h, j])).item())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_crt_predictions(
    model: CRTModel,
    test_windows_scaled: PanelWindows,
    scaler: OutcomeScaler,
    device: str,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    pred_list = []
    true_list = []

    with torch.no_grad():
        for start in range(0, len(test_windows_scaled), batch_size):
            end = min(start + batch_size, len(test_windows_scaled))
            x_hist = test_windows_scaled.x_hist[start:end].to(device)
            a_hist = test_windows_scaled.a_hist[start:end].to(device)
            y_hist = test_windows_scaled.y_hist[start:end].to(device)
            a_fut = test_windows_scaled.a_fut[start:end].to(device)
            country_idx = test_windows_scaled.country_idx[start:end].to(device)
            y_fut = test_windows_scaled.y_fut[start:end]

            y_pred = rollout(
                model=model,
                x_hist=x_hist,
                a_hist=a_hist,
                y_hist=y_hist,
                a_fut=a_fut,
                country_idx=country_idx,
            )
            pred_list.append(inverse_transform_outcomes(y_pred.cpu(), scaler))
            true_list.append(inverse_transform_outcomes(y_fut, scaler))

    return torch.cat(pred_list, dim=0), torch.cat(true_list, dim=0)


def evaluate_mean_predictions(
    test_windows_scaled: PanelWindows,
    scaler: OutcomeScaler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y_mean = torch.from_numpy(scaler.y_mean.astype(np.float32))
    n, h, d_y = test_windows_scaled.y_fut.shape
    y_pred = y_mean.view(1, 1, d_y).expand(n, h, d_y)
    y_true = inverse_transform_outcomes(test_windows_scaled.y_fut, scaler)
    return y_pred, y_true


def evaluate_persistence_predictions(
    test_windows_scaled: PanelWindows,
    scaler: OutcomeScaler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y_last = test_windows_scaled.y_hist[:, -1:, :]
    y_last_orig = inverse_transform_outcomes(y_last, scaler)
    n, _, d_y = y_last_orig.shape
    h = test_windows_scaled.y_fut.shape[1]
    y_pred = y_last_orig.expand(n, h, d_y)
    y_true = inverse_transform_outcomes(test_windows_scaled.y_fut, scaler)
    return y_pred, y_true


def build_split_windows(
    oxford_csv: str | Path,
    config_path: str | Path,
    policy_cols: List[str],
    outcome_cols: List[str],
    state_cols: List[str],
    country_to_idx: Optional[Dict[str, int]],
    scaler: Optional[OutcomeScaler] = None,
) -> Tuple[PanelWindows, PanelWindows, PanelWindows, PanelWindows, OutcomeScaler, Dict]:
    cfg = load_yaml_config(config_path)
    dataset_cfg = cfg["dataset"]
    window_cfg = cfg["window"]
    split_cfg = cfg.get("split", {})
    norm_cfg = cfg.get("normalization", {})

    raw = load_oxford_csv(oxford_csv)
    cleaned = clean_oxford(
        raw,
        country_col=dataset_cfg.get("country_col", "CountryName"),
        date_col=dataset_cfg.get("date_col", "Date"),
    )
    state_cols_sel = [s for s in state_cols if s != "__dummy_state__"]
    panel_df = select_features(
        cleaned,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols_sel,
    )

    if not state_cols_sel:
        panel_df["__dummy_state__"] = 0.0
        state_cols_sel = ["__dummy_state__"]

    if country_to_idx is None:
        country_to_idx = build_country_index(panel_df)

    windows = make_windows(
        panel_df,
        history_len=int(window_cfg["history_len"]),
        forecast_horizon=int(window_cfg["forecast_horizon"]),
        stride=int(window_cfg["stride"]),
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols_sel,
        country_to_idx=country_to_idx,
        drop_nan_windows=True,
    )

    train_idx, test_idx, cutoff_dates = time_split_window_indices(
        windows.metadata,
        train_fraction=float(split_cfg.get("train_fraction", 0.8)),
    )
    train_raw = windows.subset(train_idx)
    test_raw = windows.subset(test_idx)

    log1p = bool(norm_cfg.get("log1p_outcomes", False))
    if scaler is None:
        scaler = fit_outcome_scaler(train_raw, log1p=log1p)

    train_scaled = apply_outcome_scaler(train_raw, scaler, log1p=log1p)
    test_scaled = apply_outcome_scaler(test_raw, scaler, log1p=log1p)

    split_meta = {
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "cutoff_dates_count": int(len(cutoff_dates)),
        "log1p_outcomes": log1p,
    }
    return train_raw, test_raw, train_scaled, test_scaled, scaler, split_meta


def sample_test_windows(
    windows_raw: PanelWindows,
    windows_scaled: PanelWindows,
    max_windows: Optional[int],
) -> Tuple[PanelWindows, PanelWindows]:
    if max_windows is None or max_windows <= 0 or len(windows_raw) <= max_windows:
        return windows_raw, windows_scaled
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(windows_raw), size=max_windows, replace=False))
    return windows_raw.subset(idx), windows_scaled.subset(idx)


def compute_exact_match_stats(
    test_raw: PanelWindows,
    outcome_cols: List[str],
    atol: float = 1e-8,
) -> pd.DataFrame:
    y_last = test_raw.y_hist[:, -1:, :]
    y_true = test_raw.y_fut
    exact = torch.isclose(y_true, y_last.expand_as(y_true), atol=atol, rtol=0.0)
    exact_all = exact.all(dim=2).float().mean(dim=0).cpu().numpy()
    exact_per_outcome = exact.float().mean(dim=0).cpu().numpy()

    rows = []
    for h in range(y_true.shape[1]):
        row = {"horizon": h + 1, "exact_all_outcomes": float(exact_all[h])}
        for j, name in enumerate(outcome_cols):
            row[f"exact_{name}"] = float(exact_per_outcome[h, j])
        rows.append(row)
    return pd.DataFrame(rows)


def build_exact_match_samples(
    test_raw: PanelWindows,
    outcome_cols: List[str],
    sample_size: int = 12,
) -> pd.DataFrame:
    y_last = test_raw.y_hist[:, -1:, :]
    y_true = test_raw.y_fut
    exact = torch.isclose(y_true, y_last.expand_as(y_true), atol=1e-8, rtol=0.0)
    first4_exact_all = exact[:, :4, :].all(dim=(1, 2))
    exact_idx = torch.where(first4_exact_all)[0].cpu().numpy()

    if exact_idx.size == 0:
        chosen = np.arange(min(sample_size, len(test_raw)))
    elif exact_idx.size < sample_size:
        filler = np.setdiff1d(np.arange(len(test_raw)), exact_idx, assume_unique=False)
        needed = sample_size - exact_idx.size
        chosen = np.concatenate([exact_idx, filler[:needed]])
    else:
        chosen = exact_idx[:sample_size]

    y_last_np = y_last.squeeze(1).cpu().numpy()
    y_fut_np = y_true.cpu().numpy()
    meta = test_raw.metadata.reset_index(drop=True)

    rows: List[Dict[str, object]] = []
    for idx in chosen:
        row: Dict[str, object] = {
            "window_index": int(idx),
            "country": str(meta.iloc[idx]["country"]),
            "hist_end_date": pd.Timestamp(meta.iloc[idx]["hist_end_date"]).date().isoformat(),
            "fut_start_date": pd.Timestamp(meta.iloc[idx]["fut_start_date"]).date().isoformat(),
        }
        for j, name in enumerate(outcome_cols):
            row[f"last_{name}"] = float(y_last_np[idx, j])
            for h in range(min(4, y_fut_np.shape[1])):
                row[f"h{h+1}_{name}"] = float(y_fut_np[idx, h, j])
        row["first4_all_exact"] = bool(first4_exact_all[idx].item())
        rows.append(row)
    return pd.DataFrame(rows)


def check_split_alignment(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    scaler: OutcomeScaler,
    log1p_outcomes: bool,
) -> Dict[str, Dict]:
    train_meta = train_raw.metadata.copy()
    test_meta = test_raw.metadata.copy()

    delta_days = (test_meta["fut_start_date"] - test_meta["hist_end_date"]).dt.days
    contiguous_pass = bool((delta_days == 1).all())

    train_keys = set(
        train_meta["country"].astype(str)
        + "|"
        + train_meta["fut_start_date"].astype(str)
    )
    test_keys = set(
        test_meta["country"].astype(str)
        + "|"
        + test_meta["fut_start_date"].astype(str)
    )
    overlap = sorted(list(train_keys.intersection(test_keys)))
    overlap_pass = len(overlap) == 0

    scaler_from_train = fit_outcome_scaler(train_raw, log1p=log1p_outcomes)
    mean_match = np.allclose(scaler_from_train.y_mean, scaler.y_mean, rtol=1e-5, atol=1e-7)
    std_match = np.allclose(scaler_from_train.y_std, scaler.y_std, rtol=1e-5, atol=1e-7)
    scaler_pass = bool(mean_match and std_match)

    return {
        "contiguous_hist_to_future": {
            "pass": contiguous_pass,
            "details": {
                "expected_gap_days": 1,
                "observed_min_gap_days": int(delta_days.min()) if len(delta_days) else None,
                "observed_max_gap_days": int(delta_days.max()) if len(delta_days) else None,
            },
        },
        "train_test_no_overlap_on_country_fut_start": {
            "pass": overlap_pass,
            "details": {
                "overlap_count": len(overlap),
                "overlap_examples": overlap[:5],
            },
        },
        "scaler_consistent_with_train_only": {
            "pass": scaler_pass,
            "details": {
                "mean_match": bool(mean_match),
                "std_match": bool(std_match),
            },
        },
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.6f}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def write_stage1_report(
    report_path: Path,
    benchmark_df: pd.DataFrame,
    persistence_exact_df: pd.DataFrame,
    invariants: Dict[str, Dict],
    split_meta: Dict,
) -> Tuple[bool, str]:
    zero_like_steps = persistence_exact_df["horizon"][
        persistence_exact_df["exact_all_outcomes"] >= 0.999
    ].tolist()
    all_invariants_pass = all(v.get("pass", False) for v in invariants.values())

    if not all_invariants_pass:
        diagnosis = "Potential evaluation/splitting bug detected."
        go_no_go = "NO-GO"
    elif zero_like_steps:
        diagnosis = (
            "Early persistence near-zero error appears data-driven (carry-forward behavior) "
            "under current windowing/split, not an index overlap bug."
        )
        go_no_go = "GO"
    else:
        diagnosis = "No indexing/alignment bug detected in Stage 1 sanity checks."
        go_no_go = "GO"

    lines: List[str] = []
    lines.append("# Stage 1 Sanity Report")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- Train windows: {split_meta.get('train_count', 'n/a')}")
    lines.append(f"- Test windows: {split_meta.get('test_count', 'n/a')}")
    lines.append("")
    lines.append("## Benchmark table")
    lines.append("")
    lines.append(markdown_table(benchmark_df))
    lines.append("")
    lines.append("## Invariants")
    lines.append("")
    for name, status in invariants.items():
        mark = "PASS" if status.get("pass") else "FAIL"
        lines.append(f"- {name}: {mark}")
        details = status.get("details", {})
        for key, value in details.items():
            lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("## Persistence diagnostics")
    lines.append("")
    lines.append(
        f"- Horizons with >=99.9% exact match to last-history outcome: {zero_like_steps if zero_like_steps else 'none'}"
    )
    lines.append(f"- Interpretation: {diagnosis}")
    lines.append("")
    lines.append("## Stage 2 Gate")
    lines.append("")
    lines.append(f"- Decision: **{go_no_go}**")
    lines.append("- If GO: proceed to rollout-aware objective experiments.")
    lines.append("- If NO-GO: fix alignment/splitting/scaler issues first.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return all_invariants_pass, go_no_go


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 sanity audit for Oxford CRT benchmarks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oxford/best_crt.pt")
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/stage1")
    parser.add_argument("--max_windows", type=int, default=None, help="Use <=0 for all windows")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true", help="Save RMSE per horizon plot")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    model, _, scaler, country_to_idx, policy_cols, outcome_cols, state_cols = load_checkpoint(
        args.checkpoint, device=args.device
    )
    if not policy_cols or not outcome_cols:
        cfg = load_yaml_config(args.config)
        policy_cols = list(cfg["dataset"]["policy_cols"])
        outcome_cols = list(cfg["dataset"]["outcome_cols"])
        state_cols = list(cfg["dataset"].get("state_cols", []))

    print("Building windows and split...")
    train_raw, test_raw, train_scaled, test_scaled, scaler, split_meta = build_split_windows(
        oxford_csv=args.oxford_csv,
        config_path=args.config,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols,
        country_to_idx=country_to_idx,
        scaler=scaler,
    )

    test_raw, test_scaled = sample_test_windows(test_raw, test_scaled, args.max_windows)
    print(f"Using {len(test_scaled)} test windows for evaluation.")

    print("Evaluating CRT...")
    y_pred_crt, y_true = evaluate_crt_predictions(model, test_scaled, scaler, args.device)
    crt_rmse = compute_rmse(y_pred_crt, y_true)

    print("Evaluating mean baseline...")
    y_pred_mean, y_true_mean = evaluate_mean_predictions(test_scaled, scaler)
    mean_rmse = compute_rmse(y_pred_mean, y_true_mean)

    print("Evaluating persistence baseline...")
    y_pred_persist, y_true_persist = evaluate_persistence_predictions(test_scaled, scaler)
    persist_rmse = compute_rmse(y_pred_persist, y_true_persist)

    benchmark_rows: List[Dict[str, float | str]] = []
    baseline_rmse = mean_rmse["overall"]
    for model_name, rmse in [
        ("CRT", crt_rmse),
        ("mean", mean_rmse),
        ("persistence", persist_rmse),
    ]:
        row: Dict[str, float | str] = {
            "model": model_name,
            "overall_rmse": float(rmse["overall"]),
            "baseline_rmse": float(baseline_rmse),
            "improvement_pct": (
                100.0 * (baseline_rmse - float(rmse["overall"])) / baseline_rmse if baseline_rmse > 0 else 0.0
            ),
        }
        for h, value in enumerate(rmse["per_horizon"], start=1):
            row[f"rmse_step_{h}"] = float(value)
        benchmark_rows.append(row)
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(out_dir / "oxford_benchmarks.csv", index=False)

    print("Computing diagnostics...")
    persistence_outcome_df = per_outcome_rmse(y_pred_persist, y_true_persist, outcome_cols)
    persistence_outcome_df.to_csv(out_dir / "persistence_rmse_by_outcome_horizon.csv", index=False)

    exact_df = compute_exact_match_stats(test_raw, outcome_cols)
    exact_df.to_csv(out_dir / "persistence_exact_match_by_horizon.csv", index=False)

    samples_df = build_exact_match_samples(test_raw, outcome_cols, sample_size=12)
    samples_df.to_csv(out_dir / "persistence_sample_windows.csv", index=False)

    invariants = check_split_alignment(
        train_raw=train_raw,
        test_raw=test_raw,
        scaler=scaler,
        log1p_outcomes=bool(split_meta.get("log1p_outcomes", False)),
    )
    with open(out_dir / "stage1_invariants.json", "w", encoding="utf-8") as f:
        json.dump(invariants, f, indent=2)

    report_path = out_dir / "stage1_sanity_report.md"
    _, gate = write_stage1_report(
        report_path=report_path,
        benchmark_df=benchmark_df[["model", "overall_rmse", "baseline_rmse", "improvement_pct"]],
        persistence_exact_df=exact_df,
        invariants=invariants,
        split_meta=split_meta,
    )

    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        horizon = len(crt_rmse["per_horizon"])
        ax.plot(range(1, horizon + 1), crt_rmse["per_horizon"], label="CRT", linewidth=2)
        ax.plot(range(1, horizon + 1), mean_rmse["per_horizon"], label="mean", linewidth=2)
        ax.plot(range(1, horizon + 1), persist_rmse["per_horizon"], label="persistence", linewidth=2)
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("RMSE")
        ax.set_title("Oxford benchmarks per horizon (Stage 1)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "stage1_rmse_per_horizon.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("\nStage 1 complete.")
    print(f"Artifacts written to: {out_dir}")
    print(f"Gate decision: {gate}")


if __name__ == "__main__":
    main()
