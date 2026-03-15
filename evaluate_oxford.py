#!/usr/bin/env python
"""
Evaluate Oxford-trained CRT against benchmarks (mean predictor, persistence).

Usage:
    python evaluate_oxford.py --checkpoint checkpoints/oxford/best_crt.pt --oxford_csv data/oxford/oxford_panel.csv --config src/configs/oxford_config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from crt.model import CRTModel
from crt.rollout import rollout
from src.data.normalise import OutcomeScaler, fit_outcome_scaler, inverse_transform_outcomes

# Reuse run_rq1 helpers
from run_rq1 import build_test_windows, load_checkpoint


def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """Compute overall and per-horizon RMSE."""
    B, H, d_y = y_pred.shape
    sq_err = (y_pred - y_true) ** 2
    overall = float(torch.sqrt(torch.mean(sq_err)).item())
    per_horizon = [float(torch.sqrt(torch.mean(sq_err[:, h, :])).item()) for h in range(H)]
    return {"overall": overall, "per_horizon": per_horizon}


def evaluate_crt(
    model: CRTModel,
    test_windows,
    scaler: OutcomeScaler,
    device: str,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate CRT via rollout."""
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for start in range(0, len(test_windows), batch_size):
            end = min(start + batch_size, len(test_windows))
            x_hist = test_windows.x_hist[start:end].to(device)
            a_hist = test_windows.a_hist[start:end].to(device)
            y_hist = test_windows.y_hist[start:end].to(device)
            a_fut = test_windows.a_fut[start:end].to(device)
            country_idx = test_windows.country_idx[start:end].to(device)
            y_fut = test_windows.y_fut[start:end]

            y_pred = rollout(
                model, x_hist, a_hist, y_hist, a_fut,
                country_idx=country_idx,
            )
            y_pred_list.append(inverse_transform_outcomes(y_pred.cpu(), scaler))
            y_true_list.append(inverse_transform_outcomes(y_fut, scaler))

    y_pred_all = torch.cat(y_pred_list, dim=0)
    y_true_all = torch.cat(y_true_list, dim=0)
    return compute_rmse(y_pred_all, y_true_all)


def evaluate_mean_predictor(
    test_windows,
    scaler: OutcomeScaler,
) -> Dict[str, float]:
    """Predict training-set mean for all horizon steps."""
    y_mean = torch.from_numpy(scaler.y_mean.astype(np.float32))
    N, H, d_y = test_windows.y_fut.shape
    y_pred = y_mean.unsqueeze(0).unsqueeze(0).expand(N, H, d_y)
    y_true = inverse_transform_outcomes(test_windows.y_fut, scaler)
    return compute_rmse(y_pred, y_true)


def evaluate_persistence(
    test_windows,
    scaler: OutcomeScaler,
) -> Dict[str, float]:
    """Repeat last y_hist value for all horizon steps."""
    y_last = test_windows.y_hist[:, -1:, :]  # (N, 1, d_y) normalized
    y_last_orig = inverse_transform_outcomes(y_last, scaler)  # original scale
    N, _, d_y = y_last_orig.shape
    H = test_windows.y_fut.shape[1]
    y_pred = y_last_orig.expand(N, H, d_y)
    y_true = inverse_transform_outcomes(test_windows.y_fut, scaler)
    return compute_rmse(y_pred, y_true)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Oxford CRT vs benchmarks")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oxford/best_crt.pt", help="CRT checkpoint")
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv", help="Oxford panel CSV")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml", help="Config YAML")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--max_windows", type=int, default=None, help="Max test windows (0 = all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true", help="Save RMSE per horizon plot")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    model, config, scaler, country_to_idx, policy_cols, outcome_cols, state_cols = load_checkpoint(
        args.checkpoint, device=args.device
    )
    if not policy_cols or not outcome_cols:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        policy_cols = list(cfg["dataset"]["policy_cols"])
        outcome_cols = list(cfg["dataset"]["outcome_cols"])
        state_cols = list(cfg["dataset"].get("state_cols", []))

    print("Building test windows...")
    test_windows, scaler = build_test_windows(
        args.oxford_csv, args.config,
        policy_cols, outcome_cols, state_cols, country_to_idx, scaler=scaler
    )
    n_test = len(test_windows)
    if args.max_windows and n_test > args.max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_test, size=args.max_windows, replace=False)
        test_windows = test_windows.subset(idx)
    print(f"Evaluating on {len(test_windows)} test windows")

    results: List[Dict] = []

    # CRT
    print("Evaluating CRT...")
    crt_rmse = evaluate_crt(model, test_windows, scaler, args.device)
    results.append({"model": "CRT", "overall_rmse": crt_rmse["overall"], "per_horizon": crt_rmse["per_horizon"]})

    # Mean predictor (baseline)
    print("Evaluating mean predictor...")
    mean_rmse = evaluate_mean_predictor(test_windows, scaler)
    baseline_rmse = mean_rmse["overall"]
    results.append({"model": "mean", "overall_rmse": mean_rmse["overall"], "per_horizon": mean_rmse["per_horizon"]})

    # Persistence
    print("Evaluating persistence...")
    persist_rmse = evaluate_persistence(test_windows, scaler)
    results.append({"model": "persistence", "overall_rmse": persist_rmse["overall"], "per_horizon": persist_rmse["per_horizon"]})

    # Add baseline_rmse and improvement to each
    for r in results:
        r["baseline_rmse"] = baseline_rmse
        if baseline_rmse > 0:
            r["improvement_pct"] = 100 * (baseline_rmse - r["overall_rmse"]) / baseline_rmse
        else:
            r["improvement_pct"] = 0.0

    # Console table
    print("\n" + "=" * 70)
    print("OXFORD BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Model':<14} {'Overall RMSE':<14} {'Baseline RMSE':<14} {'Improvement':<12}")
    print("-" * 70)
    for r in results:
        imp = f"{r['improvement_pct']:+.2f}%"
        print(f"{r['model']:<14} {r['overall_rmse']:<14.4f} {r['baseline_rmse']:<14.4f} {imp:<12}")
    print("=" * 70)

    best = min(results, key=lambda x: x["overall_rmse"])
    print(f"\nBest model: {best['model']} (RMSE: {best['overall_rmse']:.4f})")

    # CSV output
    rows = []
    for r in results:
        row = {"model": r["model"], "overall_rmse": r["overall_rmse"], "baseline_rmse": r["baseline_rmse"], "improvement_pct": r["improvement_pct"]}
        for h, v in enumerate(r["per_horizon"], start=1):
            row[f"rmse_step_{h}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "oxford_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        H = len(results[0]["per_horizon"])
        for r in results:
            ax.plot(range(1, H + 1), r["per_horizon"], marker="o", label=r["model"], linewidth=2, markersize=6)
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE per forecast horizon — Oxford panel benchmarks")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "rmse_per_horizon_oxford.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
