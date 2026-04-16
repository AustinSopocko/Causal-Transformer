#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_LABELS = {
    "c00": "PRT-Control",
    "c01": "PRT-Anchor0.02",
    "c02": "PRT-Anchor0.05",
    "c03": "PRT",
    "prt": "PRT",
    "c04": "PRT-Anchor0.05+Huber",
    "c05": "PRT-Log1p",
    "persistence": "Persistence",
    "mean": "Global Mean",
    "last_7day_mean": "Last-7-Day Mean",
    "seasonal_naive_7d": "Seasonal Naive (7d)",
    "linear_trend": "Linear Trend",
    "ar_ridge_lag": "AR Ridge",
    "arx_ridge_lag_policy": "ARX Ridge (Policy)",
    "lstm_seq2seq": "LSTM Seq2Seq",
}

MODEL_ALIASES = {
    "c03": "prt",
    "prt": "c03",
}


def _label(model_id: str) -> str:
    return MODEL_LABELS.get(str(model_id), str(model_id))


def _resolve_model_key(all_df: pd.DataFrame, model_id: str) -> str:
    models = set(all_df["model"].astype(str).tolist())
    if model_id in models:
        return model_id
    alt = MODEL_ALIASES.get(model_id)
    if alt is not None and alt in models:
        return alt
    return model_id


def write_markdown_table(df: pd.DataFrame, out_path: Path, precision: int = 4) -> None:
    display = df.copy()
    for c in display.columns:
        if pd.api.types.is_numeric_dtype(display[c]):
            display[c] = display[c].map(lambda v: "nan" if pd.isna(v) else f"{float(v):.{precision}f}")
        else:
            display[c] = display[c].astype(str)

    cols = list(display.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in display.iterrows():
        body.append("| " + " | ".join([str(row[c]) for c in cols]) + " |")
    out_path.write_text("\n".join([header, sep] + body) + "\n", encoding="utf-8")


def build_main_table(h2h_dir: Path) -> pd.DataFrame:
    all_df = pd.read_csv(h2h_dir / "all_metrics_full.csv")
    short_df = pd.read_csv(h2h_dir / "short_term_metrics.csv")
    long_df = pd.read_csv(h2h_dir / "long_term_metrics.csv")
    pvsp_df = pd.read_csv(h2h_dir / "planning_vs_persistence.csv")

    main = all_df[["model", "overall_rmse", "overall_mae"]].merge(
        short_df[["model", "short_rmse_avg", "short_mae_avg"]],
        on="model",
        how="inner",
    ).merge(
        long_df[
            [
                "model",
                "long_rmse_avg",
                "long_mae_avg",
                "late_horizon_rmse",
                "trajectory_cum_mae",
                "peak_timing_mae_days",
                "peak_magnitude_mae",
                "policy_subset_rmse",
                "policy_subset_late_rmse",
                "policy_subset_n",
            ]
        ],
        on="model",
        how="inner",
    )
    keep_cols = [
        "model",
        "delta_long_rmse_vs_persistence",
        "delta_late_rmse_vs_persistence",
        "late_win_rate_vs_persistence",
        "delta_policy_subset_late_rmse_vs_persistence",
        "policy_subset_late_win_rate_vs_persistence",
    ]
    main = main.merge(pvsp_df[[c for c in keep_cols if c in pvsp_df.columns]], on="model", how="left")
    main.insert(1, "model_display", main["model"].map(_label))
    return main.sort_values("late_horizon_rmse").reset_index(drop=True)


def plot_horizon_profile(
    all_df: pd.DataFrame,
    out_path: Path,
    model_order: List[str],
    *,
    show_daily_markers: bool = True,
) -> None:
    rmse_cols = [c for c in all_df.columns if c.startswith("rmse_h")]
    if not rmse_cols:
        raise ValueError("No per-horizon RMSE columns found (rmse_h*).")
    rmse_cols = sorted(rmse_cols, key=lambda s: int(s.split("rmse_h")[-1]))
    x = [int(c.split("rmse_h")[-1]) for c in rmse_cols]

    colors = {
        "c03": "#1d4ed8",
        "persistence": "#6b7280",
        "ar_ridge_lag": "#f59e0b",
        "lstm_seq2seq": "#059669",
    }

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    for m in model_order:
        row = all_df[all_df["model"] == m]
        if row.empty:
            continue
        y = row.iloc[0][rmse_cols].astype(float).to_numpy()
        ax.plot(
            x,
            y,
            linewidth=2.5,
            label=_label(m),
            color=colors.get(m, None),
        )
        if show_daily_markers:
            ax.scatter(
                x,
                y,
                s=10,
                alpha=0.65,
                color=colors.get(m, None),
            )
    ax.set_title("Horizon Profile (RMSE by Forecast Day, h1-h42)")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("RMSE")
    ticks = [1, 6, 11, 16, 21, 26, 31, 36, 42]
    ax.set_xticks(ticks)
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_delta(
    all_df: pd.DataFrame,
    out_path: Path,
    *,
    focus_model: str,
    compare_models: List[str],
) -> None:
    rmse_cols = [c for c in all_df.columns if c.startswith("rmse_h")]
    if not rmse_cols:
        raise ValueError("No per-horizon RMSE columns found (rmse_h*).")
    rmse_cols = sorted(rmse_cols, key=lambda s: int(s.split("rmse_h")[-1]))
    x = np.array([int(c.split("rmse_h")[-1]) for c in rmse_cols], dtype=int)

    focus_resolved = _resolve_model_key(all_df, focus_model)
    row_focus = all_df[all_df["model"] == focus_resolved]
    if row_focus.empty:
        raise ValueError(f"Focus model missing from metrics: {focus_model}")
    y_focus = row_focus.iloc[0][rmse_cols].astype(float).to_numpy()

    color_map = {
        "persistence": "#6b7280",
        "ar_ridge_lag": "#f59e0b",
        "lstm_seq2seq": "#059669",
    }

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.axhline(0.0, color="#111827", linewidth=1.2, linestyle="--", alpha=0.8)

    for m in compare_models:
        m_resolved = _resolve_model_key(all_df, m)
        row_m = all_df[all_df["model"] == m_resolved]
        if row_m.empty:
            continue
        y_m = row_m.iloc[0][rmse_cols].astype(float).to_numpy()
        delta = y_focus - y_m
        ax.plot(
            x,
            delta,
            linewidth=2.3,
            label=f"{_label(focus_model)} - {_label(m)}",
            color=color_map.get(m, None),
        )
        ax.scatter(x, delta, s=9, alpha=0.6, color=color_map.get(m, None))

    ax.set_title("Daily Horizon Delta (negative = focus model better)")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("RMSE difference")
    ticks = [1, 6, 11, 16, 21, 26, 31, 36, 42]
    ax.set_xticks(ticks)
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regime_bars(
    regime_df: pd.DataFrame,
    out_path: Path,
    focus_model: str,
    reference_model: str,
) -> None:
    regime_df = regime_df[regime_df["model"].isin([focus_model, reference_model])].copy()
    if regime_df.empty:
        raise ValueError("No regime rows for requested models.")

    regime_types = list(regime_df["regime_type"].dropna().unique())
    n = len(regime_types)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    pretty_titles = {
        "outcome_level": "Case Level",
        "past_slope": "Recent Trend",
        "past_variance": "Recent Variance",
        "policy_change_magnitude": "Policy Change Magnitude",
    }

    for i, rtype in enumerate(regime_types):
        ax = axes_flat[i]
        sub = regime_df[regime_df["regime_type"] == rtype].copy()
        bins = sorted(sub["regime_bin"].dropna().unique().tolist())
        x = np.arange(len(bins))
        width = 0.36
        for j, m in enumerate([reference_model, focus_model]):
            vals: List[float] = []
            for b in bins:
                row = sub[(sub["model"] == m) & (sub["regime_bin"] == b)]
                vals.append(float(row["late_rmse"].iloc[0]) if not row.empty else np.nan)
            offset = (j - 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=_label(m),
                color="#9ca3af" if m == reference_model else "#1d4ed8",
            )
        ax.set_title(f"{pretty_titles.get(rtype, rtype)} (late RMSE)")
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=20, ha="right")
        ax.set_ylabel("Late RMSE")
        ax.grid(axis="y", alpha=0.25)

    for k in range(n, len(axes_flat)):
        axes_flat[k].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_peak_timing(
    long_df: pd.DataFrame,
    out_path: Path,
    model_order: List[str],
) -> None:
    sub = long_df[long_df["model"].isin(model_order)].copy()
    if sub.empty:
        raise ValueError("No matching models for peak timing plot.")
    sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
    sub = sub.sort_values("model")

    x = np.arange(len(sub))
    y = sub["peak_timing_mae_days"].astype(float).to_numpy()
    color_map = {
        "prt": "#1d4ed8",
        "c03": "#1d4ed8",
        "persistence": "#6b7280",
        "ar_ridge_lag": "#f59e0b",
        "lstm_seq2seq": "#059669",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, y, color=[color_map.get(str(m), "#1f77b4") for m in sub["model"]])
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in sub["model"].tolist()], rotation=20, ha="right")
    ax.set_ylabel("Peak Timing MAE (days)")
    ax.set_title("Peak Timing Error Comparison")
    # Truncated axis improves readability for close bars (state this in caption).
    y_min = max(0.0, float(np.min(y) - 3.0))
    y_max = float(np.max(y) + 1.0)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_errorbars(per_seed_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["late_horizon_rmse", "policy_subset_late_rmse"]
    means = [float(per_seed_df[m].mean()) for m in metrics]
    stds = [float(per_seed_df[m].std(ddof=1)) if len(per_seed_df) > 1 else 0.0 for m in metrics]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#1f77b4", "#ff7f0e"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Late RMSE", "Policy-Late RMSE"])
    ax.set_ylabel("Metric Value")
    ax.set_title("Seed Stability (mean ± std)")
    ax.grid(axis="y", alpha=0.25)

    rng = np.random.default_rng(42)
    for i, m in enumerate(metrics):
        vals = per_seed_df[m].astype(float).to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#111111", s=28, zorder=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage4 report-ready tables/figures from existing outputs.")
    ap.add_argument("--h2h_dir", type=str, default="results/stage4_h35_week46_h2h_with_baselines")
    ap.add_argument(
        "--h2h_lstm_dir",
        type=str,
        default="results/stage4_h35_week46_h2h_with_baselines_lstm_ctx",
    )
    ap.add_argument(
        "--horizon_source_csv",
        type=str,
        default="results/stage4_h35_week46_eval_stride1/prt_vs_baselines/all_metrics_full.csv",
    )
    ap.add_argument("--seed_dir", type=str, default="results/stage4_seed_stability")
    ap.add_argument("--focus_model", type=str, default="c03")
    ap.add_argument("--reference_model", type=str, default="persistence")
    ap.add_argument("--ar_model", type=str, default="ar_ridge_lag")
    args = ap.parse_args()

    h2h_dir = Path(args.h2h_dir)
    h2h_lstm_dir = Path(args.h2h_lstm_dir)
    seed_dir = Path(args.seed_dir)
    h2h_dir.mkdir(parents=True, exist_ok=True)

    main_df = build_main_table(h2h_dir)
    main_csv = h2h_dir / "table_main_results.csv"
    main_md = h2h_dir / "table_main_results.md"
    main_df.to_csv(main_csv, index=False)
    write_markdown_table(main_df, main_md)

    policy_df = pd.read_csv(h2h_dir / "planning_vs_persistence.csv").sort_values(
        "delta_policy_subset_late_rmse_vs_persistence"
    )
    pol_csv = h2h_dir / "table_policy_change_subset.csv"
    pol_md = h2h_dir / "table_policy_change_subset.md"
    policy_df.to_csv(pol_csv, index=False)
    write_markdown_table(policy_df, pol_md)

    regime_path = h2h_dir / "table_regime_breakdown.csv"
    if not regime_path.exists():
        regime_path = h2h_dir / "regime_breakdown_metrics.csv"
    regime_df = pd.read_csv(regime_path)
    regime_csv = h2h_dir / "table_regime_breakdown.csv"
    regime_md = h2h_dir / "table_regime_breakdown.md"
    regime_df.to_csv(regime_csv, index=False)
    write_markdown_table(regime_df, regime_md)

    regime_fig = h2h_dir / "fig_regime_late_rmse_bars.png"
    plot_regime_bars(
        regime_df=regime_df,
        out_path=regime_fig,
        focus_model=args.focus_model,
        reference_model=args.reference_model,
    )

    long_df = pd.read_csv(h2h_dir / "long_term_metrics.csv")
    model_order = [m for m in [args.focus_model, args.reference_model, args.ar_model, "lstm_seq2seq"] if m in set(long_df["model"])]
    if model_order:
        peak_fig = h2h_dir / "fig_peak_timing_mae.png"
        plot_peak_timing(long_df=long_df, out_path=peak_fig, model_order=model_order)

    all_src_override = Path(args.horizon_source_csv)
    all_src = all_src_override if all_src_override.exists() else (h2h_lstm_dir / "all_metrics_full.csv")
    if all_src.exists():
        all_df_lstm = pd.read_csv(all_src)
        horizon_fig = h2h_dir / "fig_horizon_profile_core_models.png"
        horizon_order = []
        for m in [args.focus_model, args.reference_model, args.ar_model, "lstm_seq2seq"]:
            mr = _resolve_model_key(all_df_lstm, m)
            if mr in set(all_df_lstm["model"]) and mr not in horizon_order:
                horizon_order.append(mr)
        if horizon_order:
            plot_horizon_profile(
                all_df=all_df_lstm,
                out_path=horizon_fig,
                model_order=horizon_order,
                show_daily_markers=True,
            )
            plot_horizon_delta(
                all_df=all_df_lstm,
                out_path=h2h_dir / "fig_horizon_profile_core_models_delta.png",
                focus_model=args.focus_model,
                compare_models=[args.reference_model, args.ar_model],
            )

    per_seed_path = seed_dir / "seed_stability_per_seed.csv"
    if per_seed_path.exists():
        per_seed_df = pd.read_csv(per_seed_path)
        seed_fig = seed_dir / "fig_seed_stability_errorbars.png"
        plot_seed_errorbars(per_seed_df=per_seed_df, out_path=seed_fig)

    print("Generated:")
    print(f"  - {main_csv}")
    print(f"  - {main_md}")
    print(f"  - {pol_csv}")
    print(f"  - {pol_md}")
    print(f"  - {regime_csv}")
    print(f"  - {regime_md}")
    print(f"  - {regime_fig}")
    if (h2h_dir / "fig_horizon_profile_core_models.png").exists():
        print(f"  - {h2h_dir / 'fig_horizon_profile_core_models.png'}")
    if (h2h_dir / 'fig_peak_timing_mae.png').exists():
        print(f"  - {h2h_dir / 'fig_peak_timing_mae.png'}")
    if (seed_dir / 'fig_seed_stability_errorbars.png').exists():
        print(f"  - {seed_dir / 'fig_seed_stability_errorbars.png'}")


if __name__ == "__main__":
    main()
