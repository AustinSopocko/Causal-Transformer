#!/usr/bin/env python
"""
RQ1: Policy timing and intensity — produce timing curves.

Usage:
    python run_rq1.py --checkpoint checkpoints/oxford/best_crt.pt --oxford_csv data/oxford/oxford_panel.csv --config src/configs/oxford_config.yaml [--max_windows 1000] [--output_dir results/rq1]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from crt.config import CRTConfig
from crt.model import CRTModel
from crt.rollout import rollout
from src.data.normalise import OutcomeScaler, apply_outcome_scaler, fit_outcome_scaler, inverse_transform_outcomes
from src.data.oxford_loader import (
    clean_oxford,
    load_country_context_csv,
    load_oxford_csv,
    merge_country_context,
    select_features,
)
from src.data.panel_windows import PanelWindows, build_country_index, make_windows
from src.train.splits import time_split_window_indices


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Tuple[CRTModel, CRTConfig, Optional[OutcomeScaler], Optional[Dict], List[str], List[str], List[str]]:
    """Load model, config, scaler, and column names from Oxford checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    config = ckpt.get("config")
    if config is None:
        raise ValueError("Checkpoint must contain 'config'")

    scaler = None
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler = ckpt["scaler"]
    elif "norm_stats" in ckpt and ckpt["norm_stats"] is not None:
        ns = ckpt["norm_stats"]
        scaler = OutcomeScaler(
            y_mean=ns.y_mean.numpy() if hasattr(ns.y_mean, "numpy") else np.array(ns.y_mean),
            y_std=ns.y_std.numpy() if hasattr(ns.y_std, "numpy") else np.array(ns.y_std),
        )

    model = CRTModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    policy_cols = ckpt.get("policy_cols", [])
    outcome_cols = ckpt.get("outcome_cols", [])
    state_cols = ckpt.get("state_cols", [])

    return model, config, scaler, ckpt.get("country_to_idx"), policy_cols, outcome_cols, state_cols


def build_test_windows(
    oxford_csv: str | Path,
    config_path: str | Path,
    policy_cols: List[str],
    outcome_cols: List[str],
    state_cols: List[str],
    country_to_idx: Optional[Dict[str, int]],
    train_fraction: float = 0.8,
    scaler: Optional[OutcomeScaler] = None,
) -> Tuple[PanelWindows, OutcomeScaler]:
    """Build test windows using same pipeline as training."""
    cfg = load_yaml_config(config_path)
    dataset_cfg = cfg["dataset"]
    window_cfg = cfg["window"]
    split_cfg = cfg.get("split", {})
    norm_cfg = cfg.get("normalization", {})
    context_cols = list(dataset_cfg.get("context_cols", []))
    context_csv = dataset_cfg.get("country_context_csv", None)
    split_no_future_overlap = bool(split_cfg.get("no_future_overlap", False))

    raw = load_oxford_csv(oxford_csv)
    cleaned = clean_oxford(
        raw,
        country_col=dataset_cfg.get("country_col", "CountryName"),
        date_col=dataset_cfg.get("date_col", "Date"),
        country_code_col=dataset_cfg.get("country_code_col", "CountryCode"),
    )
    if context_cols:
        if not context_csv:
            raise ValueError(
                "dataset.context_cols is non-empty but dataset.country_context_csv is not set in config."
            )
        context_df = load_country_context_csv(context_csv)
        cleaned, _ = merge_country_context(
            panel_df=cleaned,
            context_df=context_df,
            context_cols=context_cols,
            panel_country_col="country",
            panel_country_code_col="country_code",
            context_country_col=dataset_cfg.get("context_country_col", "country"),
            context_country_code_col=dataset_cfg.get("context_country_code_col", "CountryCode"),
            zscore=bool(dataset_cfg.get("context_zscore", True)),
        )
        state_cols = [*state_cols, *context_cols]
    state_cols = list(dict.fromkeys(state_cols))
    state_cols_sel = [s for s in state_cols if s != "__dummy_state__"]
    panel_df = select_features(cleaned, policy_cols=policy_cols, outcome_cols=outcome_cols, state_cols=state_cols_sel)

    if not state_cols_sel:
        panel_df["__dummy_state__"] = 0.0
        state_cols = ["__dummy_state__"]

    if country_to_idx is None:
        country_to_idx = build_country_index(panel_df)

    history_len = int(window_cfg["history_len"])
    forecast_horizon = int(window_cfg["forecast_horizon"])
    stride = int(window_cfg["stride"])

    windows = make_windows(
        panel_df,
        history_len=history_len,
        forecast_horizon=forecast_horizon,
        stride=stride,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols,
        country_to_idx=country_to_idx,
        drop_nan_windows=True,
    )

    train_idx, test_idx, _ = time_split_window_indices(
        windows.metadata,
        train_fraction=float(split_cfg.get("train_fraction", train_fraction)),
        no_future_overlap=split_no_future_overlap,
    )

    train_windows = windows.subset(train_idx)
    test_windows = windows.subset(test_idx)

    log1p = bool(norm_cfg.get("log1p_outcomes", False))
    if scaler is None:
        scaler = fit_outcome_scaler(train_windows, log1p=log1p)
    test_windows = apply_outcome_scaler(test_windows, scaler, log1p=log1p)

    return test_windows, scaler


def assign_epidemic_stage(
    y_hist: torch.Tensor,
    cases_col_idx: int = 0,
    tail_days: int = 7,
) -> np.ndarray:
    """Assign low/medium/high stage from mean cases in tail of history. Returns 0,1,2."""
    tail = min(tail_days, y_hist.shape[1])
    tail_cases = y_hist[:, -tail:, cases_col_idx].numpy()
    mean_cases = np.nanmean(tail_cases, axis=1)
    q33 = np.nanpercentile(mean_cases, 33)
    q66 = np.nanpercentile(mean_cases, 66)
    stage = np.zeros(len(mean_cases), dtype=np.int64)
    stage[mean_cases > q33] = 1
    stage[mean_cases > q66] = 2
    return stage


def build_alternative_a_fut(
    a_fut: torch.Tensor,
    delay_weeks: int,
    intensity_scale: float,
    baseline_scale: float = 0.5,
) -> torch.Tensor:
    """
    Build alternative future policy path:
    - First delay_weeks: baseline (lower) level
    - From delay_weeks onward: observed path scaled by intensity_scale
    """
    alt = a_fut.clone()
    obs_mean = a_fut.mean(dim=1, keepdim=True)
    baseline = obs_mean * baseline_scale
    if delay_weeks > 0 and delay_weeks < alt.shape[1]:
        alt[:, :delay_weeks, :] = baseline.expand(-1, delay_weeks, -1)
    alt = alt * intensity_scale
    return alt


def run_rq1(
    checkpoint_path: str | Path,
    oxford_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path = "results/rq1",
    max_windows: int = 1000,
    delay_weeks_list: Optional[List[int]] = None,
    intensity_list: Optional[List[float]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    delay_weeks_list = delay_weeks_list or [0, 2, 4, 6, 8]
    intensity_list = intensity_list or [0.8, 1.0, 1.2]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    model, config, scaler, country_to_idx, policy_cols, outcome_cols, state_cols = load_checkpoint(
        checkpoint_path, device=device
    )

    if not policy_cols or not outcome_cols:
        cfg = load_yaml_config(config_path)
        policy_cols = list(cfg["dataset"]["policy_cols"])
        outcome_cols = list(cfg["dataset"]["outcome_cols"])
        state_cols = list(cfg["dataset"].get("state_cols", []))

    print("Building test windows...")
    test_windows, scaler = build_test_windows(
        oxford_csv, config_path, policy_cols, outcome_cols, state_cols, country_to_idx, scaler=scaler
    )

    n_test = len(test_windows)
    if n_test == 0:
        raise RuntimeError("No test windows; check data and split.")
    if max_windows and n_test > max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_test, size=max_windows, replace=False)
        test_windows = test_windows.subset(idx)
        n_test = len(test_windows)
    print(f"Using {n_test} test windows")

    print("Assigning epidemic stages...")
    stages = assign_epidemic_stage(test_windows.y_hist, cases_col_idx=0)

    H = config.forecast_horizon
    d_y = config.d_y

    records: List[Dict] = []
    batch_size = 64

    for delay in delay_weeks_list:
        for intensity in intensity_list:
            print(f"  delay={delay}, intensity={intensity}")
            for start in range(0, n_test, batch_size):
                end = min(start + batch_size, n_test)
                x_hist = test_windows.x_hist[start:end].to(device)
                a_hist = test_windows.a_hist[start:end].to(device)
                y_hist = test_windows.y_hist[start:end].to(device)
                a_fut = test_windows.a_fut[start:end].to(device)
                country_idx_batch = test_windows.country_idx[start:end].to(device)

                a_alt = build_alternative_a_fut(a_fut, delay, intensity)

                with torch.no_grad():
                    y_pred = rollout(
                        model, x_hist, a_hist, y_hist, a_alt,
                        country_idx=country_idx_batch,
                    )

                if scaler is not None:
                    y_pred = inverse_transform_outcomes(y_pred.cpu(), scaler)
                else:
                    y_pred = y_pred.cpu()

                for i in range(end - start):
                    cumulative_cases = float(y_pred[i, :, 0].sum().item())
                    wid = start + i
                    records.append({
                        "window_id": wid,
                        "country": test_windows.metadata.iloc[wid]["country"],
                        "stage": ["low", "medium", "high"][int(stages[wid])],
                        "delay_weeks": delay,
                        "intensity": intensity,
                        "cumulative_cases": cumulative_cases,
                    })

    df = pd.DataFrame(records)
    df["cumulative_cases"] = df["cumulative_cases"].clip(lower=0)

    agg = df.groupby(["stage", "delay_weeks", "intensity"])["cumulative_cases"].agg(["mean", "std"]).reset_index()

    # Compute % of baseline (delay=0, intensity=1.0) so small differences are visible
    agg["pct_of_baseline"] = np.nan
    for stage in ["low", "medium", "high"]:
        sub = agg[agg["stage"] == stage]
        baseline = sub[(sub["delay_weeks"] == 0) & (sub["intensity"] == 1.0)]["mean"]
        if len(baseline) > 0 and baseline.iloc[0] > 0:
            base_val = baseline.iloc[0]
            agg.loc[agg["stage"] == stage, "pct_of_baseline"] = 100 * agg.loc[agg["stage"] == stage, "mean"] / base_val
        else:
            agg.loc[agg["stage"] == stage, "pct_of_baseline"] = 100

    # Distinct style per intensity: color, linestyle, marker
    intensity_styles = {
        0.8: {"color": "#2e86ab", "linestyle": "--", "marker": "o"},
        1.0: {"color": "#1b4332", "linestyle": "-", "marker": "s"},
        1.2: {"color": "#c73e1d", "linestyle": "-.", "marker": "^"},
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, sharey=False)
    stage_names = ["low", "medium", "high"]

    for ax_idx, stage in enumerate(stage_names):
        ax = axes[ax_idx]
        sub = agg[agg["stage"] == stage]
        for intensity in intensity_list:
            row = sub[sub["intensity"] == intensity]
            if len(row) == 0:
                continue
            row = row.sort_values("delay_weeks")
            style = intensity_styles.get(intensity, {"color": "gray", "linestyle": "-", "marker": "o"})
            ax.plot(
                row["delay_weeks"],
                row["pct_of_baseline"],
                **style,
                linewidth=2.5,
                markersize=8,
                label=f"intensity {intensity}×",
            )
        ax.axhline(100, color="gray", linestyle=":", alpha=0.7, linewidth=1)
        ax.set_xlabel("Delay (weeks)" if ax_idx == 2 else "")
        ax.set_ylabel("% of baseline (delay=0, 1.0×)")
        ax.set_title(f"Epidemic stage: {stage}", fontsize=13)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.4)
        lo, hi = sub["pct_of_baseline"].min(), sub["pct_of_baseline"].max()
        span = max(hi - lo, 3)  # minimum 3 pct points for visibility
        margin = span * 0.15
        ax.set_ylim(lo - margin, hi + margin)

    plt.suptitle("RQ1: Policy timing and intensity — predicted cumulative cases (% of immediate 1.0× baseline)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "timing_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'timing_curves.png'}")

    df.to_csv(out_dir / "results_rq1.csv", index=False)
    agg.to_csv(out_dir / "results_rq1_summary.csv", index=False)
    print(f"Saved {out_dir / 'results_rq1.csv'} and {out_dir / 'results_rq1_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1: Policy timing and intensity")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oxford/best_crt.pt", help="Oxford CRT checkpoint")
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv", help="Oxford panel CSV")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml", help="Oxford config YAML")
    parser.add_argument("--output_dir", type=str, default="results/rq1", help="Output directory")
    parser.add_argument("--max_windows", type=int, default=1000, help="Max test windows (0 = all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_rq1(
        checkpoint_path=args.checkpoint,
        oxford_csv=args.oxford_csv,
        config_path=args.config,
        output_dir=args.output_dir,
        max_windows=args.max_windows or None,
        device=args.device,
    )


if __name__ == "__main__":
    main()
