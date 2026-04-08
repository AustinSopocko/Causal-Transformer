#!/usr/bin/env python3
"""
Policy-indicator sensitivity analysis for Oxford CRT checkpoints.

Outputs:
- Tornado plot: per-policy effect size for stricter (+delta) vs looser (-delta)
- Small-multiples: full horizon delta trajectories per policy
- Per-indicator scenario fan plots for top-k most sensitive policies
- CSV/JSON artifacts with reproducible metrics and selected windows
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from evaluate_oxford_extended import (
    build_raw_train_test_windows,
    filter_negligible_countries,
    load_yaml_config,
    subset_windows,
)
from plot_oxford_scenario_fan import (
    _compute_history_variance_score,
    _compute_policy_change_score,
    _predict_with_scenario_policies,
    _select_windows,
)
from run_rq1 import load_checkpoint


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _policy_bounds(train_raw, test_raw, policy_idx: int) -> Tuple[float, float]:
    vals = torch.cat(
        [
            train_raw.a_hist[:, :, policy_idx].reshape(-1),
            train_raw.a_fut[:, :, policy_idx].reshape(-1),
            test_raw.a_hist[:, :, policy_idx].reshape(-1),
            test_raw.a_fut[:, :, policy_idx].reshape(-1),
        ],
        dim=0,
    )
    return float(torch.min(vals).item()), float(torch.max(vals).item())


def _policy_slug(name: str) -> str:
    s = name.lower().strip()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("_")
    s2 = "".join(out)
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2.strip("_")


def _policy_display_name(name: str) -> str:
    m = {
        "StringencyIndex_Average": "Stringency Index",
        "C1M_School closing": "C1M School closing",
        "C2M_Workplace closing": "C2M Workplace closing",
        "C3M_Cancel public events": "C3M Public events",
        "C4M_Restrictions on gatherings": "C4M Gatherings",
        "C6M_Stay at home requirements": "C6M Stay at home",
        "H6M_Facial Coverings": "H6M Facial coverings",
    }
    return m.get(name, name)


def _build_indicator_scenarios(
    a_fut_obs: torch.Tensor,
    policy_idx: int,
    delta_fraction: float,
    min_bound: float,
    max_bound: float,
    policy_slug: str,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {"observed": a_fut_obs.clone()}
    plus = a_fut_obs.clone()
    minus = a_fut_obs.clone()
    plus[:, :, policy_idx] = torch.clamp(
        plus[:, :, policy_idx] * (1.0 + float(delta_fraction)),
        min=min_bound,
        max=max_bound,
    )
    minus[:, :, policy_idx] = torch.clamp(
        minus[:, :, policy_idx] * (1.0 - float(delta_fraction)),
        min=min_bound,
        max=max_bound,
    )
    pct = int(round(float(delta_fraction) * 100.0))
    out[f"{policy_slug}_plus_{pct}pct"] = plus
    out[f"{policy_slug}_minus_{pct}pct"] = minus
    return out


def _save_scenario_fan_plot(
    out_path: Path,
    y_true: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    outcome_idx: int,
    outcome_name: str,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> None:
    plt = _get_pyplot()
    x = np.arange(1, int(y_true.shape[1]) + 1)

    def _band_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.quantile(arr, 0.50, axis=0)
        lo = np.quantile(arr, float(low_q), axis=0)
        hi = np.quantile(arr, float(high_q), axis=0)
        return med, lo, hi

    y_true_np = y_true.detach().cpu().numpy().astype(np.float64)[:, :, outcome_idx]
    true_med, true_lo, true_hi = _band_stats(y_true_np)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, true_lo, true_hi, color="#6b7280", alpha=0.20, linewidth=0.0)
    ax.plot(x, true_med, color="#111111", linewidth=2.2, label="truth_observed_median")

    color_map = {
        "observed": "#1f77b4",
    }
    for k in preds:
        if k.endswith("plus_10pct"):
            color_map[k] = "#d62728"
        elif k.endswith("minus_10pct"):
            color_map[k] = "#2ca02c"

    for scenario_name in preds:
        arr = preds[scenario_name].detach().cpu().numpy().astype(np.float64)[:, :, outcome_idx]
        med, lo, hi = _band_stats(arr)
        color = color_map.get(scenario_name, "#9467bd")
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0.0)
        ax.plot(x, med, color=color, linewidth=2.0, label=scenario_name)

    ax.set_title(
        f"Scenario Fan Plot ({outcome_name})\n"
        f"median + [{low_q:.0%}, {high_q:.0%}] band across selected windows"
    )
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Outcome (original scale)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tornado(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str,
    title: str,
) -> None:
    plt = _get_pyplot()
    plot_df = df.sort_values(by=f"abs_{metric_col}", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    vals = plot_df[metric_col].to_numpy(dtype=np.float64)
    labels = plot_df["policy_display"].astype(str).tolist()
    colors = np.where(vals <= 0.0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#111111", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted case difference (stricter minus looser)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    max_abs = max(1e-9, float(np.max(np.abs(vals))))
    pad = max_abs * 0.03
    for i, v in enumerate(vals):
        x_text = v + pad if v >= 0 else v - pad
        ha = "left" if v >= 0 else "right"
        ax.text(x_text, i, f"{v:.2f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_small_multiples(
    horizon_x: np.ndarray,
    trajectories: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    plt = _get_pyplot()
    policies = list(trajectories.keys())
    n = len(policies)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for i, policy in enumerate(policies):
        ax = axes_flat[i]
        rec = trajectories[policy]
        ax.fill_between(horizon_x, rec["q10"], rec["q90"], color="#1f77b4", alpha=0.16, linewidth=0.0)
        ax.plot(horizon_x, rec["mean"], color="#1f77b4", linewidth=2.0)
        ax.axhline(0.0, color="#111111", linewidth=1.0, alpha=0.7)
        ax.set_title(policy, fontsize=10)
        ax.set_xlabel("Forecast day")
        ax.set_ylabel("Delta cases (+10% minus -10%)")
        ax.grid(alpha=0.25)

    for k in range(n, len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle("Policy Indicator Sensitivity Across Horizon (cases)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy-indicator sensitivity and per-indicator scenario fan plots.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    ap.add_argument("--config", type=str, default="src/configs/stage4_h35_week46/03_huber_anchor02_nonneg.yaml")
    ap.add_argument("--output_dir", type=str, default="results/report_pack_stage4_2026-04-22/policy_sensitivity")
    ap.add_argument("--figures_dir", type=str, default="results/report_pack_stage4_2026-04-22/figures")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--window_selector", choices=["policy_top", "random"], default="policy_top")
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--delta_fraction", type=float, default=0.10)
    ap.add_argument("--late_start", type=int, default=32)
    ap.add_argument("--tornado_metric", choices=["late", "day42"], default="late")
    ap.add_argument("--top_k_fans", type=int, default=3)
    ap.add_argument("--clip_nonnegative", action="store_true")
    ap.add_argument("--save_deaths", action="store_true")
    ap.add_argument("--drop_negligible_countries", action="store_true")
    ap.add_argument("--negligible_outcome", type=str, default="new_cases_smoothed_per_million")
    ap.add_argument("--negligible_history_len", type=int, default=21)
    ap.add_argument("--negligible_threshold", type=float, default=5.0)
    ap.add_argument("--focus_high_var_policy_windows", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model, _, scaler, country_to_idx, policy_cols_ckpt, outcome_cols_ckpt, state_cols_ckpt = load_checkpoint(
        args.checkpoint, device=args.device
    )
    if (not policy_cols_ckpt) or (not outcome_cols_ckpt):
        cfg = load_yaml_config(args.config)
        policy_cols = list(cfg["dataset"]["policy_cols"])
        outcome_cols = list(cfg["dataset"]["outcome_cols"])
        state_cols = list(cfg["dataset"].get("state_cols", []))
    else:
        policy_cols = list(policy_cols_ckpt)
        outcome_cols = list(outcome_cols_ckpt)
        state_cols = list(state_cols_ckpt)

    train_raw, test_raw, log1p_outcomes = build_raw_train_test_windows(
        oxford_csv=args.oxford_csv,
        config_path=args.config,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols,
        country_to_idx=country_to_idx,
    )
    train_raw, test_raw = subset_windows(
        train_raw=train_raw,
        test_raw=test_raw,
        max_windows=args.max_windows,
        seed=int(args.seed),
    )

    dropped_countries: List[str] = []
    if args.drop_negligible_countries:
        train_raw, test_raw, country_exclusion_df, dropped_countries = filter_negligible_countries(
            train_raw=train_raw,
            test_raw=test_raw,
            outcome_cols=outcome_cols,
            outcome_name=args.negligible_outcome,
            history_len=max(1, int(args.negligible_history_len)),
            threshold=float(args.negligible_threshold),
        )
        country_exclusion_df.to_csv(out_dir / "scenario_country_exclusion_summary.csv", index=False)
        (out_dir / "scenario_countries_dropped_negligible.txt").write_text(
            "\n".join(sorted(dropped_countries)) + ("\n" if dropped_countries else ""),
            encoding="utf-8",
        )

    if "new_cases_smoothed_per_million" not in outcome_cols:
        raise ValueError("Expected 'new_cases_smoothed_per_million' in outcomes.")
    cases_idx = int(outcome_cols.index("new_cases_smoothed_per_million"))
    deaths_idx = int(outcome_cols.index("new_deaths_smoothed_per_million")) if "new_deaths_smoothed_per_million" in outcome_cols else None

    policy_score = _compute_policy_change_score(test_raw.a_fut)
    hist_var_score = _compute_history_variance_score(test_raw.y_hist, outcome_idx=cases_idx)

    candidate_idx = np.arange(len(test_raw), dtype=np.int64)
    focus_thresholds = None
    if args.focus_high_var_policy_windows:
        policy_thr = float(np.quantile(policy_score, 0.75))
        var_thr = float(np.quantile(hist_var_score, 0.75))
        focus_mask = (policy_score >= policy_thr) & (hist_var_score >= var_thr)
        focused_idx = np.where(focus_mask)[0].astype(np.int64)
        if len(focused_idx) > 0:
            candidate_idx = focused_idx
        else:
            print(
                "[warn] focus_high_var_policy_windows requested, but no windows matched; "
                "falling back to all test windows."
            )
        focus_thresholds = {
            "policy_change_q75": policy_thr,
            "history_variance_q75": var_thr,
            "n_focus_candidates": int(len(focused_idx)),
        }

    candidate_meta = test_raw.metadata.iloc[candidate_idx].reset_index(drop=True)
    candidate_policy_score = policy_score[candidate_idx]
    sel_local_idx = _select_windows(
        metadata_df=candidate_meta,
        policy_score=candidate_policy_score,
        n_windows=int(args.n_windows),
        selector=args.window_selector,
        seed=int(args.seed),
    )
    sel_idx = candidate_idx[sel_local_idx]
    selected = test_raw.subset(sel_idx)
    selected_meta = selected.metadata.copy()
    selected_meta.insert(0, "window_index", sel_idx)
    selected_meta["policy_change_score"] = policy_score[sel_idx]
    selected_meta["history_variance_score"] = hist_var_score[sel_idx]
    selected_meta.to_csv(out_dir / "scenario_window_selection.csv", index=False)

    h = int(selected.y_fut.shape[1])
    late_start_idx = max(1, min(int(args.late_start), h)) - 1
    day42_idx = h - 1
    horizon_x = np.arange(1, h + 1)

    rows: List[Dict] = []
    cases_traj: Dict[str, Dict[str, np.ndarray]] = {}
    scenario_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for p_idx, p_name in enumerate(policy_cols):
        p_slug = _policy_slug(p_name)
        p_disp = _policy_display_name(p_name)
        p_min, p_max = _policy_bounds(train_raw, test_raw, p_idx)

        scenarios = _build_indicator_scenarios(
            a_fut_obs=selected.a_fut,
            policy_idx=p_idx,
            delta_fraction=float(args.delta_fraction),
            min_bound=p_min,
            max_bound=p_max,
            policy_slug=p_slug,
        )
        preds = _predict_with_scenario_policies(
            model=model,
            scaler=scaler,
            test_raw=selected,
            a_fut_by_scenario=scenarios,
            log1p_outcomes=log1p_outcomes,
            device=args.device,
            batch_size=int(args.batch_size),
            clip_nonnegative=bool(args.clip_nonnegative),
        )
        scenario_cache[p_name] = preds

        plus_key = [k for k in preds if k.endswith("plus_10pct")][0]
        minus_key = [k for k in preds if k.endswith("minus_10pct")][0]

        delta_cases = (
            preds[plus_key][:, :, cases_idx].detach().cpu().numpy().astype(np.float64)
            - preds[minus_key][:, :, cases_idx].detach().cpu().numpy().astype(np.float64)
        )
        delta_day42 = delta_cases[:, day42_idx]
        delta_late = np.mean(delta_cases[:, late_start_idx:], axis=1)
        delta_cum = np.sum(delta_cases, axis=1)

        cases_traj[p_disp] = {
            "mean": np.mean(delta_cases, axis=0),
            "q10": np.quantile(delta_cases, 0.10, axis=0),
            "q90": np.quantile(delta_cases, 0.90, axis=0),
        }

        rows.append(
            {
                "policy": p_name,
                "policy_display": p_disp,
                "policy_slug": p_slug,
                "delta_day42_mean": float(np.mean(delta_day42)),
                "delta_day42_median": float(np.median(delta_day42)),
                "delta_late_mean": float(np.mean(delta_late)),
                "delta_late_median": float(np.median(delta_late)),
                "delta_cumulative_mean": float(np.mean(delta_cum)),
                "delta_cumulative_median": float(np.median(delta_cum)),
                "delta_day42_q10": float(np.quantile(delta_day42, 0.10)),
                "delta_day42_q90": float(np.quantile(delta_day42, 0.90)),
                "delta_late_q10": float(np.quantile(delta_late, 0.10)),
                "delta_late_q90": float(np.quantile(delta_late, 0.90)),
                "n_windows": int(delta_cases.shape[0]),
                "horizon": h,
            }
        )

    sens_df = pd.DataFrame(rows)
    sens_df["abs_delta_day42_mean"] = sens_df["delta_day42_mean"].abs()
    sens_df["abs_delta_late_mean"] = sens_df["delta_late_mean"].abs()
    if args.tornado_metric == "day42":
        metric_col = "delta_day42_mean"
        abs_metric_col = "abs_delta_day42_mean"
        tornado_title = "Policy Indicator Sensitivity (cases, day 42)"
    else:
        metric_col = "delta_late_mean"
        abs_metric_col = "abs_delta_late_mean"
        tornado_title = "Policy Indicator Sensitivity (cases, days 32-42 mean)"
    sens_df = sens_df.sort_values(abs_metric_col, ascending=False).reset_index(drop=True)
    sens_df.to_csv(out_dir / "policy_sensitivity_cases.csv", index=False)

    tornado_suffix = "day42" if args.tornado_metric == "day42" else "late"
    tornado_path = fig_dir / f"fig_policy_indicator_sensitivity_tornado_cases_{tornado_suffix}.png"
    _plot_tornado(
        df=sens_df,
        out_path=tornado_path,
        metric_col=metric_col,
        title=tornado_title,
    )

    small_mult_path = fig_dir / "fig_policy_indicator_sensitivity_small_multiples_cases.png"
    _plot_small_multiples(
        horizon_x=horizon_x,
        trajectories=cases_traj,
        out_path=small_mult_path,
    )

    top_k = max(1, min(int(args.top_k_fans), len(sens_df)))
    top_policies = sens_df.sort_values(abs_metric_col, ascending=False).head(top_k)["policy"].astype(str).tolist()
    fan_outputs: List[str] = []

    for p_name in top_policies:
        p_slug = _policy_slug(p_name)
        preds = scenario_cache[p_name]
        cases_fan_path = fig_dir / f"fig_scenario_fan_cases_{p_slug}.png"
        _save_scenario_fan_plot(
            out_path=cases_fan_path,
            y_true=selected.y_fut,
            preds=preds,
            outcome_idx=cases_idx,
            outcome_name=f"new_cases_smoothed_per_million ({_policy_display_name(p_name)})",
        )
        fan_outputs.append(str(cases_fan_path))

        if args.save_deaths and deaths_idx is not None:
            deaths_fan_path = fig_dir / f"fig_scenario_fan_deaths_{p_slug}.png"
            _save_scenario_fan_plot(
                out_path=deaths_fan_path,
                y_true=selected.y_fut,
                preds=preds,
                outcome_idx=deaths_idx,
                outcome_name=f"new_deaths_smoothed_per_million ({_policy_display_name(p_name)})",
            )
            fan_outputs.append(str(deaths_fan_path))

    meta = {
        "checkpoint": str(args.checkpoint),
        "oxford_csv": str(args.oxford_csv),
        "config": str(args.config),
        "device": str(args.device),
        "policy_cols": policy_cols,
        "outcome_cols": outcome_cols,
        "delta_fraction": float(args.delta_fraction),
        "late_start_day": int(args.late_start),
        "tornado_metric": str(args.tornado_metric),
        "n_test_windows_after_filters": int(len(test_raw)),
        "n_selected_windows": int(len(selected)),
        "window_selector": str(args.window_selector),
        "focus_high_var_policy_windows": bool(args.focus_high_var_policy_windows),
        "focus_thresholds": focus_thresholds,
        "top_policies_by_abs_delta_late_mean": top_policies,
        "drop_negligible_countries": bool(args.drop_negligible_countries),
        "n_dropped_countries": int(len(dropped_countries)),
        "clip_nonnegative": bool(args.clip_nonnegative),
        "figures": {
            "tornado_cases": str(tornado_path),
            "small_multiples_cases": str(small_mult_path),
            "fan_outputs": fan_outputs,
        },
    }
    (out_dir / "policy_sensitivity_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved policy indicator sensitivity outputs:")
    print(f"  - {out_dir / 'policy_sensitivity_cases.csv'}")
    print(f"  - {out_dir / 'scenario_window_selection.csv'}")
    print(f"  - {out_dir / 'policy_sensitivity_metadata.json'}")
    print(f"  - {tornado_path}")
    print(f"  - {small_mult_path}")
    for p in fan_outputs:
        print(f"  - {p}")
    if args.drop_negligible_countries:
        print(f"  - {out_dir / 'scenario_country_exclusion_summary.csv'}")
        print(f"  - {out_dir / 'scenario_countries_dropped_negligible.txt'}")


if __name__ == "__main__":
    main()
