#!/usr/bin/env python3
"""
Scenario fan plots for Oxford CRT checkpoints.

Runs one CRT checkpoint under multiple future-policy scenarios on selected
test windows, then plots median + spread over horizon.

Default scenarios:
  - observed policy path
  - +10% StringencyIndex_Average path (clipped to observed bounds)
  - -10% StringencyIndex_Average path (clipped to observed bounds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from crt.rollout import rollout
from evaluate_oxford_extended import (
    build_raw_train_test_windows,
    filter_negligible_countries,
    load_yaml_config,
    subset_windows,
)
from run_rq1 import load_checkpoint
from src.data.normalise import apply_outcome_scaler, inverse_transform_outcomes


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _compute_policy_change_score(a_fut: torch.Tensor) -> np.ndarray:
    """
    Per-window policy-change magnitude score from future policy path.
    """
    a = a_fut.detach().cpu().numpy().astype(np.float64)
    if a.shape[1] < 2:
        return np.zeros(a.shape[0], dtype=np.float64)
    diffs = np.abs(np.diff(a, axis=1))
    return np.mean(diffs, axis=(1, 2))


def _compute_history_variance_score(
    y_hist: torch.Tensor,
    outcome_idx: int,
) -> np.ndarray:
    """
    Per-window outcome variance over history, used to focus on dynamic regimes.
    """
    y = y_hist.detach().cpu().numpy().astype(np.float64)[:, :, outcome_idx]
    return np.var(y, axis=1)


def _select_windows(
    metadata_df: pd.DataFrame,
    policy_score: np.ndarray,
    n_windows: int,
    selector: str,
    seed: int,
) -> np.ndarray:
    n_total = len(metadata_df)
    if n_total == 0:
        return np.array([], dtype=np.int64)
    n_take = int(max(1, min(int(n_windows), n_total)))
    if selector == "policy_top":
        idx = np.argsort(-policy_score)[:n_take]
    elif selector == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_take, replace=False)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    return np.sort(idx.astype(np.int64))


def _stringency_bounds(
    train_raw,
    test_raw,
    policy_idx: int,
) -> Tuple[float, float]:
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


def _build_scenario_a_fut(
    a_fut_obs: torch.Tensor,
    policy_idx: int,
    delta_fraction: float,
    min_bound: float,
    max_bound: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "observed": a_fut_obs.clone(),
    }
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
    out[f"stringency_plus_{pct}pct"] = plus
    out[f"stringency_minus_{pct}pct"] = minus
    return out


def _predict_with_scenario_policies(
    model,
    scaler,
    test_raw,
    a_fut_by_scenario: Dict[str, torch.Tensor],
    log1p_outcomes: bool,
    device: str,
    batch_size: int,
    clip_nonnegative: bool,
) -> Dict[str, torch.Tensor]:
    model.eval()
    test_scaled = apply_outcome_scaler(test_raw, scaler, log1p=log1p_outcomes)
    n = len(test_raw)
    out: Dict[str, torch.Tensor] = {}

    for scenario_name, a_fut_scenario in a_fut_by_scenario.items():
        y_pred_list: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n, int(batch_size)):
                end = min(start + int(batch_size), n)
                x_hist = test_scaled.x_hist[start:end].to(device)
                a_hist = test_scaled.a_hist[start:end].to(device)
                y_hist = test_scaled.y_hist[start:end].to(device)
                a_fut = a_fut_scenario[start:end].to(device)
                country_idx = test_scaled.country_idx[start:end].to(device)

                y_pred_scaled = rollout(
                    model=model,
                    x_hist=x_hist,
                    a_hist=a_hist,
                    y_hist=y_hist,
                    a_fut=a_fut,
                    country_idx=country_idx,
                )
                y_pred_raw = inverse_transform_outcomes(y_pred_scaled.cpu(), scaler)
                y_pred_list.append(y_pred_raw)

        y_pred = torch.cat(y_pred_list, dim=0)
        if clip_nonnegative:
            y_pred = torch.clamp(y_pred, min=0.0)
        out[scenario_name] = y_pred
    return out


def _save_fan_plot(
    out_path: Path,
    y_true: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    outcome_idx: int,
    outcome_name: str,
    low_q: float,
    high_q: float,
    scenario_band_mode: str,
    show_delta_panel: bool,
    y_robust_low_q: float,
    y_robust_high_q: float,
    dpi: int,
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

    scenario_order = []
    if "observed" in preds:
        scenario_order.append("observed")
    scenario_order.extend(sorted([k for k in preds.keys() if k.startswith("stringency_plus_")]))
    scenario_order.extend(sorted([k for k in preds.keys() if k.startswith("stringency_minus_")]))
    scenario_order.extend([k for k in preds.keys() if k not in set(scenario_order)])

    def _scenario_label(name: str) -> str:
        if name == "observed":
            return "Observed policy"
        if name.startswith("stringency_plus_"):
            pct = name.split("_plus_")[-1].replace("pct", "")
            return f"Stricter (+{pct}%)"
        if name.startswith("stringency_minus_"):
            pct = name.split("_minus_")[-1].replace("pct", "")
            return f"Looser (-{pct}%)"
        return name

    colors = {
        "observed": "#2563eb",
    }
    for name in scenario_order:
        if name.startswith("stringency_plus_"):
            colors[name] = "#dc2626"
        elif name.startswith("stringency_minus_"):
            colors[name] = "#16a34a"
    default_palette = ["#9333ea", "#f97316", "#0ea5e9"]

    if show_delta_panel:
        fig, (ax, ax_delta) = plt.subplots(
            2,
            1,
            figsize=(11.5, 6.8),
            gridspec_kw={"height_ratios": [3.2, 1.2]},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        ax_delta = None

    ax.fill_between(x, true_lo, true_hi, color="#9ca3af", alpha=0.22, linewidth=0.0, label="Truth spread")
    ax.plot(x, true_med, color="#111111", linewidth=2.8, label="Truth median")

    palette_i = 0
    observed_med = None
    for scenario in scenario_order:
        y = preds[scenario].detach().cpu().numpy().astype(np.float64)[:, :, outcome_idx]
        med, lo, hi = _band_stats(y)
        if scenario == "observed":
            observed_med = med.copy()
        color = colors.get(scenario, default_palette[palette_i % len(default_palette)])
        if scenario not in colors:
            palette_i += 1
        if scenario_band_mode == "all":
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0.0)
        ax.plot(x, med, color=color, linewidth=2.4, label=_scenario_label(scenario))

        if ax_delta is not None and observed_med is not None and scenario != "observed":
            ax_delta.plot(
                x,
                med - observed_med,
                color=color,
                linewidth=2.2,
                label=_scenario_label(scenario),
            )

    if scenario_band_mode == "truth_only":
        pass
    elif scenario_band_mode == "none":
        # remove truth band when user wants a line-only chart
        for coll in list(ax.collections):
            coll.remove()

    if ax_delta is not None:
        ax_delta.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.2)
        ax_delta.grid(alpha=0.25)
        ax_delta.set_ylabel("Delta vs\nObserved")
        ax_delta.legend(loc="best", fontsize=9, framealpha=0.95)

    all_for_ylim = [true_lo, true_hi, true_med]
    for scenario in scenario_order:
        y = preds[scenario].detach().cpu().numpy().astype(np.float64)[:, :, outcome_idx]
        med, lo, hi = _band_stats(y)
        all_for_ylim.extend([med, lo, hi])
    flat = np.concatenate([arr.reshape(-1) for arr in all_for_ylim])
    y_lo = float(np.quantile(flat, y_robust_low_q))
    y_hi = float(np.quantile(flat, y_robust_high_q))
    pad = max(1e-6, 0.08 * (y_hi - y_lo))
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title(f"Scenario Fan Plot ({outcome_name})", fontsize=13)
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Per-million outcome")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate policy-scenario fan plots for one CRT checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/stage4_h35_week46_h2h_with_baselines")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--window_selector", choices=["policy_top", "random"], default="policy_top")
    parser.add_argument("--n_windows", type=int, default=300)
    parser.add_argument("--stringency_col", type=str, default="StringencyIndex_Average")
    parser.add_argument("--delta_fraction", type=float, default=0.10)
    parser.add_argument("--clip_nonnegative", action="store_true")
    parser.add_argument("--save_deaths", action="store_true")
    parser.add_argument("--spread_low_quantile", type=float, default=0.10)
    parser.add_argument("--spread_high_quantile", type=float, default=0.90)
    parser.add_argument(
        "--scenario_band_mode",
        choices=["all", "truth_only", "none"],
        default="truth_only",
        help="Uncertainty band style: all scenarios, truth only, or none.",
    )
    parser.add_argument(
        "--show_delta_panel",
        action="store_true",
        help="Add lower panel showing scenario medians minus observed-policy median.",
    )
    parser.add_argument("--y_robust_low_quantile", type=float, default=0.02)
    parser.add_argument("--y_robust_high_quantile", type=float, default=0.98)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--drop_negligible_countries", action="store_true")
    parser.add_argument("--negligible_outcome", type=str, default="new_cases_smoothed_per_million")
    parser.add_argument("--negligible_history_len", type=int, default=21)
    parser.add_argument("--negligible_threshold", type=float, default=5.0)
    parser.add_argument(
        "--focus_high_var_policy_windows",
        action="store_true",
        help=(
            "Restrict scenario windows to the intersection of top-quartile policy-change "
            "magnitude and top-quartile history variance (computed on cases)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, scaler, country_to_idx, policy_cols, outcome_cols, state_cols = load_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    if (not policy_cols) or (not outcome_cols):
        cfg = load_yaml_config(args.config)
        policy_cols = list(cfg["dataset"]["policy_cols"])
        outcome_cols = list(cfg["dataset"]["outcome_cols"])
        state_cols = list(cfg["dataset"].get("state_cols", []))

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

    if args.drop_negligible_countries:
        train_raw, test_raw, country_exclusion_df, dropped = filter_negligible_countries(
            train_raw=train_raw,
            test_raw=test_raw,
            outcome_cols=outcome_cols,
            outcome_name=args.negligible_outcome,
            history_len=max(1, int(args.negligible_history_len)),
            threshold=float(args.negligible_threshold),
        )
        country_exclusion_df.to_csv(out_dir / "scenario_country_exclusion_summary.csv", index=False)
        (out_dir / "scenario_countries_dropped_negligible.txt").write_text(
            "\n".join(sorted(dropped)) + ("\n" if dropped else ""),
            encoding="utf-8",
        )

    if args.stringency_col not in policy_cols:
        raise ValueError(f"Stringency column '{args.stringency_col}' not in policy_cols: {policy_cols}")
    s_idx = int(policy_cols.index(args.stringency_col))

    policy_score = _compute_policy_change_score(test_raw.a_fut)
    cases_idx_for_focus = None
    if "new_cases_smoothed_per_million" in outcome_cols:
        cases_idx_for_focus = int(outcome_cols.index("new_cases_smoothed_per_million"))
    else:
        cases_idx_for_focus = 0
    history_var_score = _compute_history_variance_score(test_raw.y_hist, outcome_idx=cases_idx_for_focus)

    candidate_idx = np.arange(len(test_raw), dtype=np.int64)
    focus_thresholds = None
    if args.focus_high_var_policy_windows:
        policy_thr = float(np.quantile(policy_score, 0.75))
        var_thr = float(np.quantile(history_var_score, 0.75))
        focus_mask = (policy_score >= policy_thr) & (history_var_score >= var_thr)
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
    selected_score = policy_score[sel_idx]
    selected_history_var = history_var_score[sel_idx]

    sel_meta = selected.metadata.copy()
    sel_meta.insert(0, "window_index", sel_idx)
    sel_meta["policy_change_score"] = selected_score
    sel_meta["history_variance_score"] = selected_history_var
    sel_meta.to_csv(out_dir / "scenario_window_selection.csv", index=False)

    p_min, p_max = _stringency_bounds(train_raw=train_raw, test_raw=test_raw, policy_idx=s_idx)
    scenario_a_fut = _build_scenario_a_fut(
        a_fut_obs=selected.a_fut,
        policy_idx=s_idx,
        delta_fraction=float(args.delta_fraction),
        min_bound=p_min,
        max_bound=p_max,
    )
    preds = _predict_with_scenario_policies(
        model=model,
        scaler=scaler,
        test_raw=selected,
        a_fut_by_scenario=scenario_a_fut,
        log1p_outcomes=log1p_outcomes,
        device=args.device,
        batch_size=int(args.batch_size),
        clip_nonnegative=bool(args.clip_nonnegative),
    )

    if "new_cases_smoothed_per_million" not in outcome_cols:
        raise ValueError(f"Outcome 'new_cases_smoothed_per_million' not in {outcome_cols}")
    cases_idx = int(outcome_cols.index("new_cases_smoothed_per_million"))
    _save_fan_plot(
        out_path=out_dir / "fig_scenario_fan_cases.png",
        y_true=selected.y_fut,
        preds=preds,
        outcome_idx=cases_idx,
        outcome_name="new_cases_smoothed_per_million",
        low_q=float(args.spread_low_quantile),
        high_q=float(args.spread_high_quantile),
        scenario_band_mode=str(args.scenario_band_mode),
        show_delta_panel=bool(args.show_delta_panel),
        y_robust_low_q=float(args.y_robust_low_quantile),
        y_robust_high_q=float(args.y_robust_high_quantile),
        dpi=int(args.dpi),
    )

    if args.save_deaths and "new_deaths_smoothed_per_million" in outcome_cols:
        deaths_idx = int(outcome_cols.index("new_deaths_smoothed_per_million"))
        _save_fan_plot(
            out_path=out_dir / "fig_scenario_fan_deaths.png",
            y_true=selected.y_fut,
            preds=preds,
            outcome_idx=deaths_idx,
            outcome_name="new_deaths_smoothed_per_million",
            low_q=float(args.spread_low_quantile),
            high_q=float(args.spread_high_quantile),
            scenario_band_mode=str(args.scenario_band_mode),
            show_delta_panel=bool(args.show_delta_panel),
            y_robust_low_q=float(args.y_robust_low_quantile),
            y_robust_high_q=float(args.y_robust_high_quantile),
            dpi=int(args.dpi),
        )

    scenario_meta = {
        "checkpoint": str(args.checkpoint),
        "oxford_csv": str(args.oxford_csv),
        "config": str(args.config),
        "device": str(args.device),
        "n_test_windows_after_filters": int(len(test_raw)),
        "n_candidate_windows_after_focus_filter": int(len(candidate_idx)),
        "n_selected_windows": int(len(selected)),
        "window_selector": str(args.window_selector),
        "focus_high_var_policy_windows": bool(args.focus_high_var_policy_windows),
        "stringency_col": str(args.stringency_col),
        "delta_fraction": float(args.delta_fraction),
        "stringency_bounds": {"min": p_min, "max": p_max},
        "policy_cols": policy_cols,
        "outcome_cols": outcome_cols,
        "scenarios": list(scenario_a_fut.keys()),
        "clip_nonnegative": bool(args.clip_nonnegative),
        "drop_negligible_countries": bool(args.drop_negligible_countries),
        "spread_quantiles": [float(args.spread_low_quantile), 0.5, float(args.spread_high_quantile)],
        "scenario_band_mode": str(args.scenario_band_mode),
        "show_delta_panel": bool(args.show_delta_panel),
        "y_robust_quantiles": [float(args.y_robust_low_quantile), float(args.y_robust_high_quantile)],
        "dpi": int(args.dpi),
    }
    if focus_thresholds is not None:
        scenario_meta["focus_thresholds"] = focus_thresholds
    (out_dir / "scenario_fan_metadata.json").write_text(
        json.dumps(scenario_meta, indent=2),
        encoding="utf-8",
    )

    print(f"Saved scenario fan outputs to {out_dir}")
    print(f"  - {out_dir / 'fig_scenario_fan_cases.png'}")
    if args.save_deaths and "new_deaths_smoothed_per_million" in outcome_cols:
        print(f"  - {out_dir / 'fig_scenario_fan_deaths.png'}")
    print(f"  - {out_dir / 'scenario_window_selection.csv'}")
    print(f"  - {out_dir / 'scenario_fan_metadata.json'}")
    if args.drop_negligible_countries:
        print(f"  - {out_dir / 'scenario_country_exclusion_summary.csv'}")
        print(f"  - {out_dir / 'scenario_countries_dropped_negligible.txt'}")


if __name__ == "__main__":
    main()
