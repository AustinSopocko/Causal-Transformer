#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate_oxford_extended import build_raw_train_test_windows, filter_negligible_countries

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


def _label(model_id: str) -> str:
    return MODEL_LABELS.get(str(model_id), str(model_id))


def _load_pred_npz(path: Path) -> np.ndarray:
    obj = np.load(path)
    if "y_pred" in obj:
        return obj["y_pred"]
    if "arr_0" in obj:
        return obj["arr_0"]
    raise KeyError(f"No y_pred/arr_0 key in {path}")


def _late_rmse(y_pred: np.ndarray, y_true: np.ndarray, late_start_1b: int) -> np.ndarray:
    s = int(max(1, late_start_1b)) - 1
    err = (y_pred[:, s:, :] - y_true[:, s:, :]) ** 2
    return np.sqrt(np.mean(err, axis=(1, 2)))


def _future_slope(y_true_2d: np.ndarray) -> np.ndarray:
    n, h = y_true_2d.shape
    x = np.arange(h, dtype=np.float64)
    x_center = x - np.mean(x)
    denom = max(np.sum(x_center * x_center), 1e-8)
    return np.sum(y_true_2d * x_center.reshape(1, -1), axis=1) / denom


def _auto_select_indices(
    y_true: np.ndarray,
    y_focus: np.ndarray,
    y_ref: np.ndarray,
    policy_mask: np.ndarray,
    late_start_1b: int,
) -> List[Tuple[str, int]]:
    rmse_focus = _late_rmse(y_focus, y_true, late_start_1b)
    rmse_ref = _late_rmse(y_ref, y_true, late_start_1b)
    delta = rmse_focus - rmse_ref

    y_true_cases = y_true[:, :, 0]
    slope = _future_slope(y_true_cases)
    var = np.var(y_true_cases, axis=1)
    level = np.mean(y_true_cases, axis=1)

    used = set()

    def _pick(mask: np.ndarray, score: np.ndarray, reverse: bool = False) -> int | None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        order = np.argsort(score[idx])
        if reverse:
            order = order[::-1]
        for j in idx[order]:
            if int(j) not in used:
                used.add(int(j))
                return int(j)
        return None

    rising_cut = float(np.quantile(slope, 0.75))
    flat_cut = float(np.quantile(var, 0.25))

    choices: List[Tuple[str, int]] = []
    i1 = _pick(mask=(policy_mask & (slope >= rising_cut) & (delta < 0)), score=delta, reverse=False)
    if i1 is not None:
        choices.append(("rising_policy_focus_wins", i1))

    level_cut = float(np.quantile(level, 0.35))
    i2 = _pick(mask=((var <= flat_cut) & (delta > 0) & (level >= level_cut)), score=delta, reverse=True)
    if i2 is not None:
        choices.append(("flat_reference_wins", i2))

    i3 = _pick(mask=(policy_mask & (delta < 0)), score=delta, reverse=False)
    if i3 is not None:
        choices.append(("policy_subset_focus_wins", i3))

    i4 = _pick(mask=(delta > 0), score=delta, reverse=True)
    if i4 is not None:
        choices.append(("reference_wins_other", i4))

    if len(choices) < 4:
        fallback = np.argsort(np.abs(delta))[::-1]
        for j in fallback:
            if int(j) in used:
                continue
            used.add(int(j))
            choices.append(("fallback", int(j)))
            if len(choices) >= 4:
                break

    return choices[:4]


def _plot_indices(
    out_path: Path,
    reasons: List[str],
    indices: List[int],
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_focus: np.ndarray,
    y_ref: np.ndarray,
    y_last: np.ndarray,
    outcome_idx: int,
    outcome_name: str,
    focus_model: str,
    reference_model: str,
) -> None:
    n = len(indices)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.8 * nrows), sharex=True)
    axes_arr = np.array(axes).reshape(-1)

    t = np.arange(0, y_true.shape[1] + 1)
    focus_label = _label(focus_model)
    reference_label = _label(reference_model)

    reason_label = {
        "rising_policy_focus_wins": "Rising + policy change (PRT wins)",
        "flat_reference_wins": "Flat regime (reference wins)",
        "policy_subset_focus_wins": "Policy-change subset (PRT wins)",
        "reference_wins_other": "Other hard case (reference wins)",
        "fallback": "Selected high-contrast case",
    }

    for ax_i, (reason, idx) in enumerate(zip(reasons, indices)):
        ax = axes_arr[ax_i]
        truth = y_true[idx, :, outcome_idx]
        focus = y_focus[idx, :, outcome_idx]
        ref = y_ref[idx, :, outcome_idx]
        # Keep trajectories physically plausible in report figures.
        truth = np.maximum(truth, 0.0)
        focus = np.maximum(focus, 0.0)
        ref = np.maximum(ref, 0.0)
        y0 = float(y_last[idx, outcome_idx])
        y0 = max(y0, 0.0)

        ax.plot(t, np.concatenate([[y0], truth]), color="#111111", linewidth=2.1, label="truth")
        ax.plot(t, np.concatenate([[y0], ref]), color="#6b7280", linestyle="--", linewidth=2.2, label=reference_label)
        ax.plot(t, np.concatenate([[y0], focus]), color="#1d4ed8", linewidth=2.4, label=focus_label)

        country_txt = ""
        if "country" in metadata_df.columns:
            country_txt = str(metadata_df.iloc[idx]["country"])
        date_txt = ""
        if "fut_start_date" in metadata_df.columns:
            d = pd.Timestamp(metadata_df.iloc[idx]["fut_start_date"])
            date_txt = d.strftime("%b %Y")
        # Keep report titles human-readable; internal ids/reasons are kept in CSV output.
        if country_txt and date_txt:
            panel_title = f"{country_txt}, {date_txt}"
        elif country_txt:
            panel_title = country_txt
        else:
            panel_title = reason_label.get(reason, reason)
        ax.set_title(panel_title, fontsize=10.0)
        ax.axvline(0, color="#9ca3af", linewidth=0.8, alpha=0.6)
        ax.set_xticks([0, 7, 14, 21, 28, 35, 42])
        ax.grid(alpha=0.25)

    for k in range(n, len(axes_arr)):
        axes_arr[k].axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, framealpha=0.95)

    outcome_pretty = outcome_name.replace("new_", "").replace("_smoothed_per_million", "").replace("_", " ")
    fig.suptitle(f"Hand-picked trajectories ({outcome_pretty})", y=0.992, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create hand-picked trajectory figure from cached predictions.")
    ap.add_argument("--h2h_dir", type=str, default="results/stage4_h35_week46_h2h_with_baselines")
    ap.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    ap.add_argument("--config", type=str, default="src/configs/stage4_h35_week46/00_control_h35.yaml")
    ap.add_argument("--focus_model", type=str, default="c03")
    ap.add_argument("--reference_model", type=str, default="persistence")
    ap.add_argument("--outcome", type=str, default="new_cases_smoothed_per_million")
    ap.add_argument("--window_indices", type=str, default="", help="Comma-separated manual indices. Empty => auto-select.")
    args = ap.parse_args()

    h2h_dir = Path(args.h2h_dir)
    meta = json.loads((h2h_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
    policy_cols = list(meta["policy_cols"])
    outcome_cols = list(meta["outcome_cols"])
    late_start = int(meta.get("late_horizon_start", 32))
    if args.outcome not in outcome_cols:
        raise ValueError(f"Outcome '{args.outcome}' not found in {outcome_cols}")
    outcome_idx = outcome_cols.index(args.outcome)

    train_raw, test_raw, _ = build_raw_train_test_windows(
        oxford_csv=args.oxford_csv,
        config_path=args.config,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=[],
        country_to_idx=None,
    )

    cf = meta.get("country_filter", {})
    if bool(cf.get("enabled", False)):
        train_raw, test_raw, _, _ = filter_negligible_countries(
            train_raw=train_raw,
            test_raw=test_raw,
            outcome_cols=outcome_cols,
            outcome_name=str(cf.get("outcome", "new_cases_smoothed_per_million")),
            history_len=int(cf.get("history_len", 21)),
            threshold=float(cf.get("threshold", 5.0)),
        )

    y_true = test_raw.y_fut.detach().cpu().numpy().astype(np.float64)
    y_last = test_raw.y_hist[:, -1, :].detach().cpu().numpy().astype(np.float64)
    n = y_true.shape[0]

    focus_npz = h2h_dir / "pred_cache" / f"{args.focus_model}.npz"
    if not focus_npz.exists():
        raise FileNotFoundError(f"Missing pred cache for focus model: {focus_npz}")
    y_focus = _load_pred_npz(focus_npz).astype(np.float64)

    ref_npz = h2h_dir / "pred_cache" / f"{args.reference_model}.npz"
    if ref_npz.exists():
        y_ref = _load_pred_npz(ref_npz).astype(np.float64)
    elif args.reference_model == "persistence":
        h = y_true.shape[1]
        y_ref = np.repeat(y_last[:, None, :], repeats=h, axis=1).astype(np.float64)
    elif args.reference_model == "mean":
        mu = np.mean(y_true.reshape(-1, y_true.shape[-1]), axis=0, keepdims=True)
        y_ref = np.tile(mu.reshape(1, 1, -1), (y_true.shape[0], y_true.shape[1], 1)).astype(np.float64)
    else:
        raise FileNotFoundError(
            f"Missing pred cache for reference model and no built-in fallback: {ref_npz}"
        )

    if y_focus.shape[0] != n or y_ref.shape[0] != n:
        raise ValueError(
            f"Prediction-window mismatch: truth={n}, focus={y_focus.shape[0]}, ref={y_ref.shape[0]}."
        )

    policy_mask_df = pd.read_csv(h2h_dir / "policy_change_subset_windows.csv")
    policy_mask = policy_mask_df["is_policy_change_subset"].to_numpy(dtype=np.int64).astype(bool)
    if policy_mask.shape[0] != n:
        raise ValueError(f"Policy mask length mismatch: mask={policy_mask.shape[0]} truth={n}.")

    if args.window_indices.strip():
        idx = [int(v.strip()) for v in args.window_indices.split(",") if v.strip()]
        idx = [i for i in idx if 0 <= i < n]
        reasons = [f"manual_{k+1}" for k in range(len(idx))]
    else:
        picked = _auto_select_indices(
            y_true=y_true,
            y_focus=y_focus,
            y_ref=y_ref,
            policy_mask=policy_mask,
            late_start_1b=late_start,
        )
        reasons = [r for r, _ in picked]
        idx = [i for _, i in picked]

    out_png = h2h_dir / f"fig_trajectory_handpicked_{args.outcome}.png"
    _plot_indices(
        out_path=out_png,
        reasons=reasons,
        indices=idx,
        metadata_df=test_raw.metadata,
        y_true=y_true,
        y_focus=y_focus,
        y_ref=y_ref,
        y_last=y_last,
        outcome_idx=outcome_idx,
        outcome_name=args.outcome,
        focus_model=args.focus_model,
        reference_model=args.reference_model,
    )

    out_rows = []
    for reason, i in zip(reasons, idx):
        row = {
            "reason": reason,
            "window_index": int(i),
            "country": test_raw.metadata.iloc[int(i)]["country"] if "country" in test_raw.metadata.columns else "",
            "fut_start_date": (
                str(pd.Timestamp(test_raw.metadata.iloc[int(i)]["fut_start_date"]).date())
                if "fut_start_date" in test_raw.metadata.columns
                else ""
            ),
        }
        out_rows.append(row)
    out_csv = h2h_dir / "trajectory_handpicked_selection.csv"
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)

    print("Generated:")
    print(f"  - {out_png}")
    print(f"  - {out_csv}")


if __name__ == "__main__":
    main()
