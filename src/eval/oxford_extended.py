"""Extended Oxford evaluation metrics and table builders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class HorizonBuckets:
    """1-based horizon boundaries."""

    short_end: int
    long_start: int
    late_start: int
    horizon: int

    def validate(self) -> None:
        if self.horizon < 2:
            raise ValueError(f"horizon must be >= 2, got {self.horizon}")
        if not (1 <= self.short_end <= self.horizon):
            raise ValueError(f"short_end must be in [1, {self.horizon}], got {self.short_end}")
        if not (1 <= self.long_start <= self.horizon):
            raise ValueError(f"long_start must be in [1, {self.horizon}], got {self.long_start}")
        if not (1 <= self.late_start <= self.horizon):
            raise ValueError(f"late_start must be in [1, {self.horizon}], got {self.late_start}")
        if self.short_end >= self.long_start:
            raise ValueError(
                f"Expected short_end < long_start for disjoint short/long buckets. "
                f"Got short_end={self.short_end}, long_start={self.long_start}"
            )
        if self.late_start < self.long_start:
            raise ValueError(
                f"Expected late_start >= long_start. Got late_start={self.late_start}, long_start={self.long_start}"
            )


def default_horizon_buckets(horizon: int) -> HorizonBuckets:
    """
    Defaults for epidemic forecasting:
    - 42-step horizon: short 1..7, long 22..H, late 32..H
    - 28-step horizon: short 1..7, long 15..H, late 22..H
    - 14-step horizon: short 1..4, long 8..H, late 12..H
    Fallback: proportional split by horizon length.
    """
    if horizon >= 42:
        short_end = 7
        long_start = 22
        late_start = 32
    elif horizon >= 28:
        short_end = 7
        long_start = 15
        late_start = 22
    elif horizon >= 14:
        short_end = 4
        long_start = 8
        late_start = 12
    else:
        short_end = max(2, horizon // 3)
        long_start = max(short_end + 1, horizon // 2 + 1)
        late_start = max(long_start, horizon - max(3, horizon // 4) + 1)
    buckets = HorizonBuckets(
        short_end=short_end,
        long_start=long_start,
        late_start=late_start,
        horizon=horizon,
    )
    buckets.validate()
    return buckets


def compute_per_horizon_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-horizon RMSE and MAE arrays of shape (H,)."""
    abs_err = torch.abs(y_pred - y_true)
    sq_err = (y_pred - y_true) ** 2
    mae_h = torch.mean(abs_err, dim=(0, 2)).cpu().numpy()
    rmse_h = torch.sqrt(torch.mean(sq_err, dim=(0, 2))).cpu().numpy()
    return rmse_h.astype(np.float64), mae_h.astype(np.float64)


def _avg_in_range(values: np.ndarray, start_1b: int, end_1b: int) -> float:
    s = start_1b - 1
    e = end_1b
    return float(np.mean(values[s:e]))


def compute_late_horizon_rmse(rmse_h: np.ndarray, late_start_1b: int) -> float:
    return float(np.mean(rmse_h[late_start_1b - 1 :]))


def compute_cumulative_trajectory_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Whole-rollout trajectory error:
    MAE between cumulative trajectories over horizon.
    """
    cum_pred = torch.cumsum(y_pred, dim=1)
    cum_true = torch.cumsum(y_true, dim=1)
    return float(torch.mean(torch.abs(cum_pred - cum_true)).item())


def compute_peak_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    peak_outcome_idx: int,
) -> Dict[str, float]:
    """
    Peak timing and magnitude errors on one selected outcome dimension.
    timing units are horizon steps (days in this dataset).
    """
    pred_series = y_pred[:, :, peak_outcome_idx]
    true_series = y_true[:, :, peak_outcome_idx]

    pred_peak_idx = torch.argmax(pred_series, dim=1).to(torch.float32) + 1.0
    true_peak_idx = torch.argmax(true_series, dim=1).to(torch.float32) + 1.0
    peak_timing_mae = float(torch.mean(torch.abs(pred_peak_idx - true_peak_idx)).item())

    pred_peak_val = torch.max(pred_series, dim=1).values
    true_peak_val = torch.max(true_series, dim=1).values
    peak_magnitude_mae = float(torch.mean(torch.abs(pred_peak_val - true_peak_val)).item())

    return {
        "peak_timing_mae_days": peak_timing_mae,
        "peak_magnitude_mae": peak_magnitude_mae,
    }


def compute_policy_change_mask(
    a_fut: torch.Tensor,
    method: str = "quantile",
    quantile: float = 0.75,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Policy-change score and subset mask.
    score = mean absolute step-to-step future policy change over horizon and policy dims.
    """
    if a_fut.shape[1] < 2:
        score = torch.zeros(a_fut.shape[0], dtype=torch.float32).cpu().numpy()
        mask = np.zeros_like(score, dtype=bool)
        return score, mask

    delta = torch.abs(a_fut[:, 1:, :] - a_fut[:, :-1, :])
    score = torch.mean(delta, dim=(1, 2)).cpu().numpy().astype(np.float64)

    method = str(method).lower()
    if method == "quantile":
        q = float(np.clip(quantile, 0.0, 1.0))
        cut = float(np.quantile(score, q))
        mask = score >= cut
    elif method == "threshold":
        cut = float(threshold)
        mask = score >= cut
    else:
        raise ValueError(f"Unknown policy change method '{method}'. Use 'quantile' or 'threshold'.")
    return score, mask


def summarize_model_metrics(
    model_name: str,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    a_fut_raw: torch.Tensor,
    buckets: HorizonBuckets,
    policy_subset_mask: np.ndarray,
    peak_outcome_idx: int,
) -> Dict[str, float | str]:
    rmse_h, mae_h = compute_per_horizon_metrics(y_pred, y_true)
    metrics: Dict[str, float | str] = {"model": model_name}

    for h in range(1, buckets.horizon + 1):
        metrics[f"rmse_h{h}"] = float(rmse_h[h - 1])
        metrics[f"mae_h{h}"] = float(mae_h[h - 1])

    metrics["overall_rmse"] = float(np.mean(rmse_h))
    metrics["overall_mae"] = float(np.mean(mae_h))
    metrics["short_rmse_avg"] = _avg_in_range(rmse_h, 1, buckets.short_end)
    metrics["short_mae_avg"] = _avg_in_range(mae_h, 1, buckets.short_end)
    metrics["long_rmse_avg"] = _avg_in_range(rmse_h, buckets.long_start, buckets.horizon)
    metrics["long_mae_avg"] = _avg_in_range(mae_h, buckets.long_start, buckets.horizon)
    metrics["late_horizon_rmse"] = compute_late_horizon_rmse(rmse_h, buckets.late_start)
    metrics["trajectory_cum_mae"] = compute_cumulative_trajectory_mae(y_pred, y_true)

    peak = compute_peak_metrics(y_pred, y_true, peak_outcome_idx=peak_outcome_idx)
    metrics.update(peak)

    mask_t = torch.from_numpy(policy_subset_mask.astype(np.bool_))
    if int(mask_t.sum()) > 0:
        y_pred_sub = y_pred[mask_t]
        y_true_sub = y_true[mask_t]
        sub_rmse_h, _ = compute_per_horizon_metrics(y_pred_sub, y_true_sub)
        metrics["policy_subset_n"] = int(mask_t.sum().item())
        metrics["policy_subset_rmse"] = float(np.mean(sub_rmse_h))
        metrics["policy_subset_late_rmse"] = float(np.mean(sub_rmse_h[buckets.late_start - 1 :]))
    else:
        metrics["policy_subset_n"] = 0
        metrics["policy_subset_rmse"] = float("nan")
        metrics["policy_subset_late_rmse"] = float("nan")

    return metrics


def build_short_term_table(metrics_df: pd.DataFrame, buckets: HorizonBuckets) -> pd.DataFrame:
    cols: List[str] = ["model", "short_rmse_avg", "short_mae_avg"]
    for h in range(1, buckets.short_end + 1):
        cols.append(f"rmse_h{h}")
        cols.append(f"mae_h{h}")
    return metrics_df[cols].copy()


def build_long_term_table(metrics_df: pd.DataFrame, buckets: HorizonBuckets) -> pd.DataFrame:
    cols: List[str] = [
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
    for h in range(buckets.long_start, buckets.horizon + 1):
        cols.append(f"rmse_h{h}")
        cols.append(f"mae_h{h}")
    return metrics_df[cols].copy()


def highlighted_markdown_table(
    df: pd.DataFrame,
    metric_columns: Sequence[str],
    lower_is_better: bool = True,
    precision: int = 4,
) -> str:
    """
    Return markdown with best metric in each column wrapped in **...**.
    Only numeric metric columns are highlighted.
    """
    display = df.copy()
    for c in metric_columns:
        if c not in display.columns:
            continue
        values = pd.to_numeric(display[c], errors="coerce")
        if values.notna().sum() == 0:
            continue
        target = values.min() if lower_is_better else values.max()
        is_best = np.isclose(values.to_numpy(dtype=np.float64), float(target), rtol=1e-10, atol=1e-12)
        formatted = []
        for i, v in enumerate(values.to_numpy()):
            if np.isnan(v):
                formatted.append("nan")
            else:
                val_str = f"{float(v):.{precision}f}"
                formatted.append(f"**{val_str}**" if is_best[i] else val_str)
        display[c] = formatted

    for c in display.columns:
        if c not in metric_columns:
            display[c] = display[c].astype(str)

    cols = list(display.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in display.iterrows():
        vals = [str(row[c]) for c in cols]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def save_table_outputs(
    table_df: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
    metric_columns: Sequence[str],
    precision: int = 4,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(csv_path, index=False)
    md = highlighted_markdown_table(table_df, metric_columns=metric_columns, precision=precision)
    md_path.write_text(md + "\n", encoding="utf-8")


def compute_window_segment_rmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    start_1b: int,
    end_1b: Optional[int] = None,
) -> np.ndarray:
    """
    Per-window RMSE over horizon segment [start_1b, end_1b] (1-based, inclusive).
    Returns shape (N,).
    """
    h = int(y_true.shape[1])
    start = max(1, min(int(start_1b), h))
    end = h if end_1b is None else max(start, min(int(end_1b), h))
    seg = (y_pred - y_true)[:, start - 1 : end, :]
    rmse_n = torch.sqrt(torch.mean(seg * seg, dim=(1, 2))).cpu().numpy()
    return rmse_n.astype(np.float64)


def paired_win_rate(candidate_rmse: np.ndarray, reference_rmse: np.ndarray) -> float:
    """
    Pairwise win rate with ties counted as half-wins.
    """
    if candidate_rmse.shape != reference_rmse.shape:
        raise ValueError(
            f"Shape mismatch in paired_win_rate: candidate={candidate_rmse.shape}, reference={reference_rmse.shape}"
        )
    valid = ~(np.isnan(candidate_rmse) | np.isnan(reference_rmse))
    if int(np.sum(valid)) == 0:
        return float("nan")
    c = candidate_rmse[valid]
    r = reference_rmse[valid]
    wins = np.mean(c < r)
    ties = np.mean(np.isclose(c, r, rtol=1e-10, atol=1e-12))
    return float(wins + 0.5 * ties)


def build_persistence_comparison_table(
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    policy_subset_mask: np.ndarray,
    buckets: HorizonBuckets,
    reference_model: str = "persistence",
) -> pd.DataFrame:
    """
    Build planning-focused comparison table against persistence on identical windows.
    Delta metrics are candidate - persistence, so negative is better.
    """
    if reference_model not in predictions_by_model:
        raise ValueError(f"Reference model '{reference_model}' not found in predictions_by_model")

    policy_mask = policy_subset_mask.astype(bool)
    y_ref = predictions_by_model[reference_model]
    ref_long = compute_window_segment_rmse(y_ref, y_true, start_1b=buckets.long_start, end_1b=buckets.horizon)
    ref_late = compute_window_segment_rmse(y_ref, y_true, start_1b=buckets.late_start, end_1b=buckets.horizon)
    ref_policy_late = ref_late[policy_mask] if int(policy_mask.sum()) > 0 else np.array([], dtype=np.float64)

    rows: List[Dict[str, float | str]] = []
    for model_name in sorted(predictions_by_model.keys()):
        y_pred = predictions_by_model[model_name]
        cand_long = compute_window_segment_rmse(y_pred, y_true, start_1b=buckets.long_start, end_1b=buckets.horizon)
        cand_late = compute_window_segment_rmse(y_pred, y_true, start_1b=buckets.late_start, end_1b=buckets.horizon)
        cand_policy_late = cand_late[policy_mask] if int(policy_mask.sum()) > 0 else np.array([], dtype=np.float64)

        row: Dict[str, float | str] = {
            "model": model_name,
            "delta_long_rmse_vs_persistence": float(np.mean(cand_long) - np.mean(ref_long)),
            "delta_late_rmse_vs_persistence": float(np.mean(cand_late) - np.mean(ref_late)),
            "late_win_rate_vs_persistence": paired_win_rate(cand_late, ref_late),
            "policy_subset_n": int(policy_mask.sum()),
        }
        if int(policy_mask.sum()) > 0:
            row["delta_policy_subset_late_rmse_vs_persistence"] = float(
                np.mean(cand_policy_late) - np.mean(ref_policy_late)
            )
            row["policy_subset_late_win_rate_vs_persistence"] = paired_win_rate(cand_policy_late, ref_policy_late)
        else:
            row["delta_policy_subset_late_rmse_vs_persistence"] = float("nan")
            row["policy_subset_late_win_rate_vs_persistence"] = float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["delta_late_rmse_vs_persistence", "delta_long_rmse_vs_persistence"]).reset_index(drop=True)
