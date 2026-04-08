#!/usr/bin/env python
"""
Extended Oxford evaluation:
- multi-model comparison on same split
- short-term and long-term metrics tables
- late-horizon, trajectory, peak, and policy-change subset metrics
- best-in-column highlighting in markdown table outputs

Examples:
  python evaluate_oxford_extended.py \
    --model baseline_crt=checkpoints/oxford/best_crt.pt \
    --model rollout_crt=checkpoints/oxford_stage2/03_rollout_mix_25/best_crt.pt \
    --oxford_csv data/oxford/oxford_panel.csv \
    --config src/configs/oxford_config.yaml \
    --output_dir results/oxford_extended
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from crt.rollout import rollout
from run_rq1 import load_checkpoint
from src.data.normalise import apply_outcome_scaler, inverse_transform_outcomes
from src.data.oxford_loader import (
    clean_oxford,
    load_country_context_csv,
    load_oxford_csv,
    merge_country_context,
    select_features,
)
from src.data.panel_windows import PanelWindows, build_country_index, make_windows
from src.eval.oxford_extended import (
    HorizonBuckets,
    build_persistence_comparison_table,
    build_long_term_table,
    build_short_term_table,
    compute_window_segment_rmse,
    compute_policy_change_mask,
    default_horizon_buckets,
    paired_win_rate,
    save_table_outputs,
    summarize_model_metrics,
)
from src.train.splits import time_split_window_indices


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_model_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model '{spec}'. Expected format: label=/path/to/checkpoint.pt")
    label, ckpt = spec.split("=", 1)
    label = label.strip()
    ckpt = ckpt.strip()
    if not label:
        raise ValueError(f"Invalid --model '{spec}': empty label")
    if not ckpt:
        raise ValueError(f"Invalid --model '{spec}': empty checkpoint path")
    return label, ckpt


def build_raw_train_test_windows(
    oxford_csv: str | Path,
    config_path: str | Path,
    policy_cols: List[str],
    outcome_cols: List[str],
    state_cols: List[str],
    country_to_idx: Optional[Dict[str, int]],
) -> Tuple[PanelWindows, PanelWindows, bool]:
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
    train_idx, test_idx, _ = time_split_window_indices(
        windows.metadata,
        train_fraction=float(split_cfg.get("train_fraction", 0.8)),
        no_future_overlap=split_no_future_overlap,
    )
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError("Empty train/test split from Oxford windows.")
    train_raw = windows.subset(train_idx)
    test_raw = windows.subset(test_idx)
    return train_raw, test_raw, bool(norm_cfg.get("log1p_outcomes", False))


def subset_windows(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    max_windows: Optional[int],
    seed: int = 42,
) -> Tuple[PanelWindows, PanelWindows]:
    if max_windows is None or max_windows <= 0 or len(test_raw) <= max_windows:
        return train_raw, test_raw
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(test_raw), size=max_windows, replace=False))
    return train_raw, test_raw.subset(idx)


def filter_negligible_countries(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    outcome_cols: List[str],
    outcome_name: str,
    history_len: int = 21,
    threshold: float = 5.0,
) -> Tuple[PanelWindows, PanelWindows, pd.DataFrame, List[str]]:
    """
    Drop countries entirely when their test windows never exceed a minimum
    recent-incidence level.

    A country is marked negligible if:
      max(mean(y_hist over last history_len days)) < threshold
    where y_hist refers to `outcome_name` in original units.
    """
    if outcome_name not in outcome_cols:
        raise ValueError(f"negligible outcome '{outcome_name}' not in outcomes: {outcome_cols}")
    if len(test_raw) == 0:
        raise ValueError("Cannot filter negligible countries: empty test set.")

    outcome_idx = int(outcome_cols.index(outcome_name))
    hist_t = int(test_raw.y_hist.shape[1])
    h_eff = int(max(1, min(int(history_len), hist_t)))
    thr = float(threshold)

    y_hist_test = test_raw.y_hist[:, :, outcome_idx].detach().cpu().numpy().astype(np.float64)
    recent_mean_test = np.mean(y_hist_test[:, -h_eff:], axis=1)
    meta_test = test_raw.metadata.reset_index(drop=True).copy()
    meta_test["recent_mean_outcome"] = recent_mean_test

    by_country = (
        meta_test.groupby("country")["recent_mean_outcome"]
        .agg(n_test_windows="size", max_recent_mean="max", mean_recent_mean="mean", std_recent_mean="std")
        .reset_index()
    )
    by_country["std_recent_mean"] = by_country["std_recent_mean"].fillna(0.0)
    by_country["is_negligible_country"] = (by_country["max_recent_mean"] < thr).astype(int)
    dropped_countries = by_country.loc[by_country["is_negligible_country"] == 1, "country"].astype(str).tolist()
    dropped_set = set(dropped_countries)

    keep_test_mask = ~meta_test["country"].astype(str).isin(dropped_set).to_numpy()
    keep_test_idx = np.where(keep_test_mask)[0]
    if keep_test_idx.size == 0:
        raise RuntimeError(
            "Negligible-country filter removed all test windows. "
            "Lower --negligible_threshold or disable --drop_negligible_countries."
        )
    test_f = test_raw.subset(keep_test_idx)

    meta_train = train_raw.metadata.reset_index(drop=True).copy()
    keep_train_mask = ~meta_train["country"].astype(str).isin(dropped_set).to_numpy()
    keep_train_idx = np.where(keep_train_mask)[0]
    if keep_train_idx.size == 0:
        raise RuntimeError(
            "Negligible-country filter removed all train windows. "
            "Lower --negligible_threshold or disable --drop_negligible_countries."
        )
    train_f = train_raw.subset(keep_train_idx)

    train_counts = meta_train.groupby("country").size().rename("n_train_windows")
    country_summary = by_country.set_index("country").join(train_counts, how="left").fillna({"n_train_windows": 0})
    country_summary["n_train_windows"] = country_summary["n_train_windows"].astype(int)
    country_summary = country_summary.reset_index()
    country_summary["kept_country"] = (1 - country_summary["is_negligible_country"]).astype(int)
    country_summary = country_summary.sort_values(
        ["is_negligible_country", "max_recent_mean", "n_test_windows"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return train_f, test_f, country_summary, dropped_countries


def compute_mean_baseline_prediction(train_raw: PanelWindows, test_raw: PanelWindows) -> torch.Tensor:
    d_y = test_raw.y_fut.shape[2]
    train_y = torch.cat(
        [
            train_raw.y_hist.reshape(-1, d_y),
            train_raw.y_fut.reshape(-1, d_y),
        ],
        dim=0,
    )
    y_mean = torch.mean(train_y, dim=0, keepdim=True)  # (1, d_y)
    n, h, _ = test_raw.y_fut.shape
    return y_mean.view(1, 1, d_y).expand(n, h, d_y).clone()


def compute_persistence_prediction(test_raw: PanelWindows) -> torch.Tensor:
    y_last = test_raw.y_hist[:, -1:, :]  # raw scale
    n, _, d_y = y_last.shape
    h = test_raw.y_fut.shape[1]
    return y_last.expand(n, h, d_y).clone()


def compute_seasonal_naive_prediction(
    test_raw: PanelWindows,
    seasonal_period: int = 7,
) -> torch.Tensor:
    """
    Seasonal naive baseline from history.
    Repeats the last `seasonal_period` values into the future horizon.
    """
    y_hist = test_raw.y_hist  # raw scale, (N, T, d_y)
    n, t, d_y = y_hist.shape
    h = test_raw.y_fut.shape[1]
    if t <= 0:
        raise ValueError("History length must be > 0 for seasonal naive baseline.")
    period = int(max(1, min(seasonal_period, t)))
    recent = y_hist[:, -period:, :]  # (N, P, d_y)
    idx = (torch.arange(h, dtype=torch.long) % period).to(torch.long)
    y_pred = recent[:, idx, :]  # (N, H, d_y)
    return y_pred.reshape(n, h, d_y).clone()


def compute_linear_trend_prediction(test_raw: PanelWindows) -> torch.Tensor:
    """
    Linear trend baseline per window and outcome dimension.
    Fits y = a + b*t over history with least-squares and extrapolates to horizon.
    """
    y_hist = test_raw.y_hist  # raw scale, (N, T, d_y)
    n, t, d_y = y_hist.shape
    h = test_raw.y_fut.shape[1]

    if t < 2:
        return compute_persistence_prediction(test_raw)

    t_idx = torch.arange(t, dtype=y_hist.dtype)
    t_mean = torch.mean(t_idx)
    centered = t_idx - t_mean
    denom = torch.sum(centered * centered)
    if float(denom) <= 0.0:
        return compute_persistence_prediction(test_raw)

    slope = torch.sum(y_hist * centered.view(1, t, 1), dim=1) / denom  # (N, d_y)
    intercept = torch.mean(y_hist, dim=1) - slope * t_mean  # (N, d_y)

    fut_idx = torch.arange(t, t + h, dtype=y_hist.dtype)
    y_pred = intercept.unsqueeze(1) + slope.unsqueeze(1) * fut_idx.view(1, h, 1)
    return y_pred.reshape(n, h, d_y).clone()


def compute_last_k_mean_prediction(test_raw: PanelWindows, k: int = 7) -> torch.Tensor:
    """
    Last-k mean baseline.
    Predicts the mean of the most recent k observed outcome values for each window.
    """
    y_hist = test_raw.y_hist  # (N, T, d_y)
    n, t, d_y = y_hist.shape
    h = test_raw.y_fut.shape[1]
    k_eff = int(max(1, min(int(k), int(t))))
    y_ref = torch.mean(y_hist[:, -k_eff:, :], dim=1, keepdim=True)
    return y_ref.expand(n, h, d_y).clone()


def _flatten_tail_features(tensor_3d: torch.Tensor, tail_len: int) -> np.ndarray:
    """
    Convert (N, T, D) tensor to flattened tail features (N, tail_len * D).
    If tail_len > T, left-pad by repeating the first available step.
    """
    arr = tensor_3d.detach().cpu().numpy().astype(np.float32)
    n, t, d = arr.shape
    tail_len = int(max(1, tail_len))
    take = min(t, tail_len)
    tail = arr[:, -take:, :]
    if take < tail_len:
        pad = np.repeat(tail[:, :1, :], repeats=tail_len - take, axis=1)
        tail = np.concatenate([pad, tail], axis=1)
    return tail.reshape(n, tail_len * d)


def _one_hot_country(country_idx: torch.Tensor, num_classes: Optional[int] = None) -> np.ndarray:
    idx = country_idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(np.max(idx)) + 1 if idx.size > 0 else 1
    one_hot = np.zeros((idx.shape[0], int(num_classes)), dtype=np.float32)
    one_hot[np.arange(idx.shape[0]), np.clip(idx, 0, int(num_classes) - 1)] = 1.0
    return one_hot


def _build_lag_design(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    lag_len: int,
    include_future_policy: bool,
    include_country_context: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Build train/test design matrices and flattened horizon targets.
    Returns: X_train, Y_train_flat, X_test, horizon, d_y
    """
    n_train, h, d_y = train_raw.y_fut.shape
    n_test = test_raw.y_fut.shape[0]

    x_train_parts: List[np.ndarray] = [_flatten_tail_features(train_raw.y_hist, tail_len=lag_len)]
    x_test_parts: List[np.ndarray] = [_flatten_tail_features(test_raw.y_hist, tail_len=lag_len)]

    if include_future_policy:
        x_train_parts.append(train_raw.a_fut.detach().cpu().numpy().reshape(n_train, -1).astype(np.float32))
        x_test_parts.append(test_raw.a_fut.detach().cpu().numpy().reshape(n_test, -1).astype(np.float32))

    if include_country_context:
        num_classes = int(
            max(
                int(torch.max(train_raw.country_idx).item()) if len(train_raw.country_idx) > 0 else 0,
                int(torch.max(test_raw.country_idx).item()) if len(test_raw.country_idx) > 0 else 0,
            )
            + 1
        )
        x_train_parts.append(_one_hot_country(train_raw.country_idx, num_classes=num_classes))
        x_test_parts.append(_one_hot_country(test_raw.country_idx, num_classes=num_classes))

    x_train = np.concatenate(x_train_parts, axis=1).astype(np.float32)
    x_test = np.concatenate(x_test_parts, axis=1).astype(np.float32)
    y_train = train_raw.y_fut.detach().cpu().numpy().reshape(n_train, h * d_y).astype(np.float32)
    return x_train, y_train, x_test, int(h), int(d_y)


def _fit_predict_multioutput_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Closed-form multi-target ridge regression with standardized features.
    """
    x_mu = np.mean(x_train, axis=0, keepdims=True)
    x_std = np.std(x_train, axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_train_z = (x_train - x_mu) / x_std
    x_test_z = (x_test - x_mu) / x_std

    n, p = x_train_z.shape
    x_train_1 = np.concatenate([np.ones((n, 1), dtype=np.float32), x_train_z], axis=1)
    x_test_1 = np.concatenate([np.ones((x_test_z.shape[0], 1), dtype=np.float32), x_test_z], axis=1)

    reg = float(max(alpha, 0.0)) * np.eye(p + 1, dtype=np.float32)
    reg[0, 0] = 0.0  # do not regularize intercept
    xtx = x_train_1.T @ x_train_1
    xty = x_train_1.T @ y_train
    try:
        w = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xtx + reg) @ xty
    return (x_test_1 @ w).astype(np.float32)


def compute_ar_ridge_prediction(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    lag_len: int = 21,
    alpha: float = 1.0,
    include_country_context: bool = False,
) -> torch.Tensor:
    """
    AR baseline: ridge on lagged outcomes only, direct multi-horizon prediction.
    """
    x_train, y_train, x_test, h, d_y = _build_lag_design(
        train_raw=train_raw,
        test_raw=test_raw,
        lag_len=lag_len,
        include_future_policy=False,
        include_country_context=include_country_context,
    )
    y_pred_flat = _fit_predict_multioutput_ridge(x_train=x_train, y_train=y_train, x_test=x_test, alpha=alpha)
    return torch.from_numpy(y_pred_flat.reshape(x_test.shape[0], h, d_y))


def compute_arx_ridge_prediction(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    lag_len: int = 21,
    alpha: float = 1.0,
    include_country_context: bool = False,
) -> torch.Tensor:
    """
    ARX baseline: ridge on lagged outcomes + future policy sequence.
    """
    x_train, y_train, x_test, h, d_y = _build_lag_design(
        train_raw=train_raw,
        test_raw=test_raw,
        lag_len=lag_len,
        include_future_policy=True,
        include_country_context=include_country_context,
    )
    y_pred_flat = _fit_predict_multioutput_ridge(x_train=x_train, y_train=y_train, x_test=x_test, alpha=alpha)
    return torch.from_numpy(y_pred_flat.reshape(x_test.shape[0], h, d_y))


def compute_boosted_lag_prediction(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    lag_len: int = 21,
    max_train_windows: int = 6000,
    include_future_policy: bool = True,
    include_country_context: bool = False,
    backend: str = "auto",
    max_iter: int = 120,
    random_state: int = 42,
) -> Tuple[Optional[torch.Tensor], str]:
    """
    Tree-boosted lag baseline over flattened horizon targets.
    Backend order for 'auto': sklearn HistGradientBoosting -> xgboost -> lightgbm.
    Returns (prediction or None, backend_name_used_or_reason).
    """
    x_train, y_train, x_test, h, d_y = _build_lag_design(
        train_raw=train_raw,
        test_raw=test_raw,
        lag_len=lag_len,
        include_future_policy=include_future_policy,
        include_country_context=include_country_context,
    )

    if x_train.shape[0] > int(max_train_windows):
        rng = np.random.default_rng(int(random_state))
        idx = np.sort(rng.choice(x_train.shape[0], size=int(max_train_windows), replace=False))
        x_train = x_train[idx]
        y_train = y_train[idx]

    backend_req = str(backend).lower()
    backend_used = ""

    model_factory = None
    if backend_req in {"none", "off", "disable", "disabled"}:
        return None, "boosted_disabled"

    if backend_req in {"auto", "hist_gbm"}:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

            backend_used = "hist_gbm"

            def model_factory():
                return HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                )
        except Exception:
            if backend_req == "hist_gbm":
                return None, "hist_gbm_unavailable"

    if model_factory is None and backend_req in {"auto", "xgboost"}:
        try:
            from xgboost import XGBRegressor  # type: ignore

            backend_used = "xgboost"

            def model_factory():
                return XGBRegressor(
                    n_estimators=int(max_iter),
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=int(random_state),
                    n_jobs=1,
                )
        except Exception:
            if backend_req == "xgboost":
                return None, "xgboost_unavailable"

    if model_factory is None and backend_req in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMRegressor  # type: ignore

            backend_used = "lightgbm"

            def model_factory():
                return LGBMRegressor(
                    n_estimators=int(max_iter),
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=int(random_state),
                    n_jobs=1,
                    verbosity=-1,
                )
        except Exception:
            if backend_req == "lightgbm":
                return None, "lightgbm_unavailable"

    if model_factory is None:
        return None, "no_boost_backend_available"

    y_pred_flat = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=np.float32)
    for target_j in range(y_train.shape[1]):
        model = model_factory()
        model.fit(x_train, y_train[:, target_j])
        y_pred_flat[:, target_j] = model.predict(x_test).astype(np.float32)

    y_pred = torch.from_numpy(y_pred_flat.reshape(x_test.shape[0], h, d_y))
    return y_pred, backend_used


class _Seq2SeqLSTMBaseline(nn.Module):
    """
    Lightweight seq2seq LSTM baseline trained on panel windows.
    Encoder consumes history; decoder rolls out autoregressively with optional teacher forcing.
    """

    def __init__(
        self,
        d_y: int,
        d_a: int,
        d_country: int,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        enc_in = int(d_y + d_a + d_country)
        dec_in = int(d_y + d_a + d_country)
        self.encoder = nn.LSTM(input_size=enc_in, hidden_size=int(hidden_dim), num_layers=1, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=dec_in, hidden_size=int(hidden_dim))
        self.readout = nn.Linear(int(hidden_dim), int(d_y))

    def forward(
        self,
        y_hist: torch.Tensor,
        a_hist: torch.Tensor,
        a_fut: torch.Tensor,
        country_ctx: torch.Tensor,
        y_fut: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> torch.Tensor:
        n, t, _ = y_hist.shape
        h = int(a_fut.shape[1])

        if country_ctx.shape[1] > 0:
            c_hist = country_ctx.unsqueeze(1).expand(n, t, country_ctx.shape[1])
            enc_in = torch.cat([y_hist, a_hist, c_hist], dim=-1)
        else:
            enc_in = torch.cat([y_hist, a_hist], dim=-1)

        _, (h_n, c_n) = self.encoder(enc_in)
        h_t = h_n[-1]  # (N, hidden)
        c_t = c_n[-1]  # (N, hidden)

        y_prev = y_hist[:, -1, :]
        preds: List[torch.Tensor] = []
        for step in range(h):
            a_step = a_fut[:, step, :]
            if country_ctx.shape[1] > 0:
                dec_in = torch.cat([y_prev, a_step, country_ctx], dim=-1)
            else:
                dec_in = torch.cat([y_prev, a_step], dim=-1)

            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            y_hat = self.readout(h_t)
            preds.append(y_hat.unsqueeze(1))

            if y_fut is not None and teacher_forcing_prob > 0.0:
                mask = (torch.rand(y_hat.shape[0], device=y_hat.device) < float(teacher_forcing_prob)).unsqueeze(1)
                y_prev = torch.where(mask, y_fut[:, step, :], y_hat)
            else:
                y_prev = y_hat

        return torch.cat(preds, dim=1)


def _safe_standardize(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mu) / torch.clamp(std, min=1e-6)


def compute_lstm_seq2seq_prediction(
    train_raw: PanelWindows,
    test_raw: PanelWindows,
    include_future_policy: bool = True,
    include_country_context: bool = False,
    hidden_dim: int = 96,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_train_windows: int = 12000,
    val_fraction: float = 0.1,
    patience: int = 4,
    teacher_forcing_start: float = 0.9,
    teacher_forcing_end: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Train a small LSTM baseline on train windows and return raw-scale test predictions.
    """
    rng = np.random.default_rng(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    n_train = int(len(train_raw))
    if n_train == 0:
        raise ValueError("Empty train windows for LSTM baseline.")

    if n_train > int(max_train_windows):
        idx = np.sort(rng.choice(n_train, size=int(max_train_windows), replace=False))
        train_raw = train_raw.subset(idx)
        n_train = int(len(train_raw))

    d_y = int(train_raw.y_fut.shape[2])
    d_a_raw = int(train_raw.a_hist.shape[2])

    # Per-dimension standardization in raw space for numeric stability.
    y_all = torch.cat(
        [
            train_raw.y_hist.reshape(-1, d_y),
            train_raw.y_fut.reshape(-1, d_y),
        ],
        dim=0,
    ).to(torch.float32)
    y_mu = torch.mean(y_all, dim=0, keepdim=True)
    y_std = torch.std(y_all, dim=0, keepdim=True, unbiased=False)

    if include_future_policy and d_a_raw > 0:
        a_all = torch.cat(
            [
                train_raw.a_hist.reshape(-1, d_a_raw),
                train_raw.a_fut.reshape(-1, d_a_raw),
            ],
            dim=0,
        ).to(torch.float32)
        a_mu = torch.mean(a_all, dim=0, keepdim=True)
        a_std = torch.std(a_all, dim=0, keepdim=True, unbiased=False)
        d_a_model = d_a_raw
    else:
        a_mu = torch.zeros((1, 1), dtype=torch.float32)
        a_std = torch.ones((1, 1), dtype=torch.float32)
        d_a_model = 0

    def _prep_a_hist(raw: PanelWindows) -> torch.Tensor:
        if d_a_model == 0:
            return torch.zeros((len(raw), raw.a_hist.shape[1], 0), dtype=torch.float32)
        return _safe_standardize(raw.a_hist.to(torch.float32), a_mu.view(1, 1, -1), a_std.view(1, 1, -1))

    def _prep_a_fut(raw: PanelWindows) -> torch.Tensor:
        if d_a_model == 0:
            return torch.zeros((len(raw), raw.a_fut.shape[1], 0), dtype=torch.float32)
        return _safe_standardize(raw.a_fut.to(torch.float32), a_mu.view(1, 1, -1), a_std.view(1, 1, -1))

    def _prep_country_ctx(raw: PanelWindows) -> torch.Tensor:
        if not include_country_context:
            return torch.zeros((len(raw), 0), dtype=torch.float32)
        max_id = int(
            max(
                int(torch.max(train_raw.country_idx).item()) if len(train_raw.country_idx) > 0 else 0,
                int(torch.max(test_raw.country_idx).item()) if len(test_raw.country_idx) > 0 else 0,
            )
        )
        num_classes = max_id + 1
        one_hot = _one_hot_country(raw.country_idx, num_classes=num_classes)
        return torch.from_numpy(one_hot).to(torch.float32)

    y_hist_train = _safe_standardize(train_raw.y_hist.to(torch.float32), y_mu.view(1, 1, -1), y_std.view(1, 1, -1))
    y_fut_train = _safe_standardize(train_raw.y_fut.to(torch.float32), y_mu.view(1, 1, -1), y_std.view(1, 1, -1))
    a_hist_train = _prep_a_hist(train_raw)
    a_fut_train = _prep_a_fut(train_raw)
    c_train = _prep_country_ctx(train_raw)

    perm = rng.permutation(n_train)
    n_val = int(max(1, round(n_train * float(np.clip(val_fraction, 0.0, 0.5))))) if n_train >= 10 else 0
    if n_val >= n_train:
        n_val = max(0, n_train - 1)
    val_idx = np.sort(perm[:n_val]) if n_val > 0 else np.array([], dtype=np.int64)
    tr_idx = np.sort(perm[n_val:]) if n_val > 0 else np.arange(n_train, dtype=np.int64)

    def _idx_tensor(x: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
        if idx.size == 0:
            return x[:0]
        return x[torch.from_numpy(idx).to(torch.long)]

    train_ds = TensorDataset(
        _idx_tensor(y_hist_train, tr_idx),
        _idx_tensor(a_hist_train, tr_idx),
        _idx_tensor(a_fut_train, tr_idx),
        _idx_tensor(y_fut_train, tr_idx),
        _idx_tensor(c_train, tr_idx),
    )
    train_loader = DataLoader(train_ds, batch_size=int(max(8, batch_size)), shuffle=True, drop_last=False)

    val_tensors = (
        _idx_tensor(y_hist_train, val_idx),
        _idx_tensor(a_hist_train, val_idx),
        _idx_tensor(a_fut_train, val_idx),
        _idx_tensor(y_fut_train, val_idx),
        _idx_tensor(c_train, val_idx),
    )

    run_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = _Seq2SeqLSTMBaseline(
        d_y=d_y,
        d_a=d_a_model,
        d_country=int(c_train.shape[1]),
        hidden_dim=int(max(16, hidden_dim)),
    ).to(run_device)
    optim = torch.optim.Adam(model.parameters(), lr=float(lr))
    mse = nn.MSELoss()

    best_state: Optional[dict] = None
    best_val = float("inf")
    epochs_no_improve = 0
    n_epochs = int(max(1, epochs))

    for epoch in range(n_epochs):
        model.train()
        if n_epochs <= 1:
            tf_prob = float(teacher_forcing_end)
        else:
            progress = float(epoch) / float(n_epochs - 1)
            tf_prob = float(teacher_forcing_start + (teacher_forcing_end - teacher_forcing_start) * progress)
        tf_prob = float(np.clip(tf_prob, 0.0, 1.0))

        for y_hist_b, a_hist_b, a_fut_b, y_fut_b, c_b in train_loader:
            y_hist_b = y_hist_b.to(run_device)
            a_hist_b = a_hist_b.to(run_device)
            a_fut_b = a_fut_b.to(run_device)
            y_fut_b = y_fut_b.to(run_device)
            c_b = c_b.to(run_device)

            optim.zero_grad(set_to_none=True)
            y_hat_b = model(
                y_hist=y_hist_b,
                a_hist=a_hist_b,
                a_fut=a_fut_b,
                country_ctx=c_b,
                y_fut=y_fut_b,
                teacher_forcing_prob=tf_prob,
            )
            loss = mse(y_hat_b, y_fut_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        if val_idx.size == 0:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        model.eval()
        with torch.no_grad():
            y_hist_v, a_hist_v, a_fut_v, y_fut_v, c_v = [t.to(run_device) for t in val_tensors]
            y_hat_v = model(
                y_hist=y_hist_v,
                a_hist=a_hist_v,
                a_fut=a_fut_v,
                country_ctx=c_v,
                y_fut=None,
                teacher_forcing_prob=0.0,
            )
            val_loss = float(mse(y_hat_v, y_fut_v).item())

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(max(1, patience)):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_hist_test = _safe_standardize(test_raw.y_hist.to(torch.float32), y_mu.view(1, 1, -1), y_std.view(1, 1, -1)).to(run_device)
        a_hist_test = _prep_a_hist(test_raw).to(run_device)
        a_fut_test = _prep_a_fut(test_raw).to(run_device)
        c_test = _prep_country_ctx(test_raw).to(run_device)
        y_pred_scaled = model(
            y_hist=y_hist_test,
            a_hist=a_hist_test,
            a_fut=a_fut_test,
            country_ctx=c_test,
            y_fut=None,
            teacher_forcing_prob=0.0,
        )
        y_pred_raw = y_pred_scaled.cpu() * y_std.view(1, 1, -1) + y_mu.view(1, 1, -1)
    return y_pred_raw.to(torch.float32)


def _country_index_match_or_raise(
    reference: Optional[Dict[str, int]],
    current: Optional[Dict[str, int]],
    label: str,
) -> None:
    if reference is None or current is None:
        return
    if reference != current:
        raise ValueError(
            f"Country index mapping mismatch for model '{label}'. "
            "For fair model-to-model evaluation, country mappings must be identical."
        )


def predict_crt_from_checkpoint(
    model_label: str,
    checkpoint_path: str | Path,
    test_raw: PanelWindows,
    device: str,
    log1p_outcomes: bool,
    reference_country_to_idx: Optional[Dict[str, int]],
    cache_dir: Optional[Path],
    use_cache: bool = True,
    batch_size: int = 64,
    allow_horizon_truncate: bool = False,
) -> torch.Tensor:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{model_label}.npz"

    if use_cache and cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        y_pred = torch.from_numpy(cached["y_pred"]).to(torch.float32)
        return y_pred

    model, config, scaler, country_to_idx, _, _, _ = load_checkpoint(checkpoint_path, device=device)
    if scaler is None:
        raise ValueError(f"Checkpoint '{checkpoint_path}' has no scaler; cannot run Oxford evaluation.")
    _country_index_match_or_raise(reference_country_to_idx, country_to_idx, model_label)

    if config.forecast_horizon != test_raw.y_fut.shape[1]:
        if not allow_horizon_truncate:
            raise ValueError(
                f"Horizon mismatch for '{model_label}': checkpoint horizon={config.forecast_horizon}, "
                f"test horizon={test_raw.y_fut.shape[1]}. "
                "If you intentionally want to evaluate a longer-horizon checkpoint on a shorter-horizon "
                "test setup, pass --allow_horizon_truncate."
            )
        print(
            f"[warn] Horizon mismatch for '{model_label}': checkpoint horizon={config.forecast_horizon}, "
            f"test horizon={test_raw.y_fut.shape[1]}. "
            "Proceeding because --allow_horizon_truncate is enabled."
        )

    test_scaled = apply_outcome_scaler(test_raw, scaler, log1p=log1p_outcomes)

    y_pred_list: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_scaled), batch_size):
            end = min(start + batch_size, len(test_scaled))
            x_hist = test_scaled.x_hist[start:end].to(device)
            a_hist = test_scaled.a_hist[start:end].to(device)
            y_hist = test_scaled.y_hist[start:end].to(device)
            a_fut = test_scaled.a_fut[start:end].to(device)
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

    if cache_path is not None:
        np.savez_compressed(cache_path, y_pred=y_pred.numpy())

    return y_pred


def write_metadata(
    out_path: Path,
    buckets: HorizonBuckets,
    policy_change_method: str,
    policy_change_quantile: float,
    policy_change_threshold: float,
    clip_nonnegative: bool,
    policy_subset_n: int,
    policy_subset_ratio: float,
    model_labels: List[str],
    outcome_cols: List[str],
    policy_cols: List[str],
    country_filter: Optional[dict] = None,
) -> None:
    meta = {
        "horizon": buckets.horizon,
        "short_horizon_end": buckets.short_end,
        "long_horizon_start": buckets.long_start,
        "late_horizon_start": buckets.late_start,
        "policy_change_method": policy_change_method,
        "policy_change_quantile": policy_change_quantile,
        "policy_change_threshold": policy_change_threshold,
        "clip_nonnegative": bool(clip_nonnegative),
        "policy_subset_n": policy_subset_n,
        "policy_subset_ratio": policy_subset_ratio,
        "models": model_labels,
        "outcome_cols": outcome_cols,
        "policy_cols": policy_cols,
    }
    if country_filter is not None:
        meta["country_filter"] = country_filter
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_plain_markdown_table(df: pd.DataFrame, out_path: Path, precision: int = 4) -> None:
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
        vals = [str(row[c]) for c in cols]
        body.append("| " + " | ".join(vals) + " |")
    out_path.write_text("\n".join([header, sep] + body) + "\n", encoding="utf-8")


def _rmse_horizon_cols(metrics_df: pd.DataFrame) -> List[str]:
    cols: List[Tuple[int, str]] = []
    for c in metrics_df.columns:
        if c.startswith("rmse_h"):
            try:
                idx = int(c.replace("rmse_h", ""))
                cols.append((idx, c))
            except ValueError:
                continue
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_horizon_profile_plot(metrics_df: pd.DataFrame, out_path: Path) -> bool:
    plt = _get_pyplot()

    rmse_cols = _rmse_horizon_cols(metrics_df)
    if not rmse_cols:
        return False

    x = np.arange(1, len(rmse_cols) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for _, row in metrics_df.iterrows():
        model = str(row["model"])
        y = [float(row[c]) for c in rmse_cols]
        lw = 2.0 if model in {"persistence", "mean"} else 1.5
        ls = "--" if model == "persistence" else "-"
        alpha = 0.95 if model.startswith("c") else 0.85
        ax.plot(x, y, label=model, linewidth=lw, linestyle=ls, alpha=alpha)

    ax.set_title("RMSE by Forecast Horizon")
    ax.set_xlabel("Horizon Day")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    ax.set_xlim(1, len(rmse_cols))
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build_main_results_table(
    metrics_df: pd.DataFrame,
    short_df: pd.DataFrame,
    long_df: pd.DataFrame,
    persistence_cmp_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    overall_cols = ["model"]
    if "overall_rmse" in metrics_df.columns:
        overall_cols.append("overall_rmse")
    if "overall_mae" in metrics_df.columns:
        overall_cols.append("overall_mae")

    main = metrics_df[overall_cols].merge(
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

    if persistence_cmp_df is not None and not persistence_cmp_df.empty:
        keep_cols = [
            "model",
            "delta_long_rmse_vs_persistence",
            "delta_late_rmse_vs_persistence",
            "late_win_rate_vs_persistence",
            "delta_policy_subset_late_rmse_vs_persistence",
            "policy_subset_late_win_rate_vs_persistence",
        ]
        cmp = persistence_cmp_df[[c for c in keep_cols if c in persistence_cmp_df.columns]].copy()
        main = main.merge(cmp, on="model", how="left")

    return main.sort_values("late_horizon_rmse").reset_index(drop=True)


def _short_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def save_trajectory_diagnostics(
    output_dir: Path,
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    y_last_obs: Optional[torch.Tensor],
    metadata_df: pd.DataFrame,
    buckets: HorizonBuckets,
    focus_model: str,
    reference_model: str,
    outcome_idx: int,
    outcome_name: str,
    samples_per_group: int = 20,
    seed: int = 42,
    handpicked_indices: Optional[np.ndarray] = None,
) -> List[Path]:
    plt = _get_pyplot()

    if focus_model not in predictions_by_model:
        print(f"[trajectory] focus model '{focus_model}' not found; skipping plots.")
        return []
    if reference_model not in predictions_by_model:
        print(f"[trajectory] reference model '{reference_model}' not found; skipping plots.")
        return []

    focus_rmse = compute_window_segment_rmse(
        predictions_by_model[focus_model], y_true, start_1b=buckets.late_start, end_1b=buckets.horizon
    )
    ref_rmse = compute_window_segment_rmse(
        predictions_by_model[reference_model], y_true, start_1b=buckets.late_start, end_1b=buckets.horizon
    )
    better_idx = np.where(focus_rmse < ref_rmse)[0]
    worse_idx = np.where(focus_rmse > ref_rmse)[0]

    rng = np.random.default_rng(seed)

    def _sample(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return idx
        n = min(int(samples_per_group), int(idx.size))
        chosen = rng.choice(idx, size=n, replace=False)
        return np.sort(chosen)

    better_sel = _sample(better_idx)
    worse_sel = _sample(worse_idx)

    baseline_names = {
        "mean",
        "persistence",
        "seasonal_naive_7d",
        "linear_trend",
        "last_7day_mean",
        "ar_ridge_lag",
        "arx_ridge_lag_policy",
        "boosted_lag_policy",
        "lstm_seq2seq",
    }
    crt_variants = [m for m in sorted(predictions_by_model.keys()) if m not in baseline_names]
    lines_order = [reference_model] + [m for m in crt_variants if m != reference_model]
    colors = {
        reference_model: "#6b7280",
        "truth": "#111111",
    }
    palette = plt.get_cmap("tab10")
    for i, model_name in enumerate([m for m in lines_order if m != reference_model]):
        colors[model_name] = palette(i % 10)

    saved_paths: List[Path] = []

    def _plot_group(indices: np.ndarray, group_name: str) -> Optional[Path]:
        if indices.size == 0:
            print(f"[trajectory] no windows in group '{group_name}'")
            return None
        n = int(indices.size)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.4 * nrows), sharex=True)
        axes_arr = np.array(axes).reshape(-1)
        include_t0 = y_last_obs is not None
        t = np.arange(0, y_true.shape[1] + 1) if include_t0 else np.arange(1, y_true.shape[1] + 1)

        for ax_i, win_idx in enumerate(indices):
            ax = axes_arr[ax_i]
            truth = y_true[win_idx, :, outcome_idx].cpu().numpy()
            y0 = (
                y_last_obs[win_idx, outcome_idx].detach().cpu().numpy()
                if include_t0
                else None
            )
            truth_plot = np.concatenate([np.array([float(y0)]), truth]) if include_t0 else truth
            ax.plot(t, truth_plot, color=colors["truth"], linewidth=2.0, label="truth")

            ref_pred = predictions_by_model[reference_model][win_idx, :, outcome_idx].cpu().numpy()
            ref_plot = np.concatenate([np.array([float(y0)]), ref_pred]) if include_t0 else ref_pred
            ax.plot(t, ref_plot, color=colors[reference_model], linestyle="--", linewidth=1.7, label=reference_model)

            for model_name in [m for m in lines_order if m != reference_model]:
                pred = predictions_by_model[model_name][win_idx, :, outcome_idx].cpu().numpy()
                pred_plot = np.concatenate([np.array([float(y0)]), pred]) if include_t0 else pred
                ax.plot(t, pred_plot, color=colors[model_name], linewidth=1.3, alpha=0.95, label=model_name)

            title_bits = [f"idx={int(win_idx)}"]
            if "country" in metadata_df.columns:
                title_bits.append(str(metadata_df.iloc[int(win_idx)]["country"]))
            if "fut_start_date" in metadata_df.columns:
                title_bits.append(str(pd.Timestamp(metadata_df.iloc[int(win_idx)]["fut_start_date"]).date()))
            ax.set_title(" | ".join(title_bits), fontsize=8)
            if include_t0:
                ax.axvline(0, color="#9ca3af", linewidth=0.8, alpha=0.6)
                ax.set_xticks([0, 7, 14, 21, 28, 35, 42])
            ax.grid(alpha=0.2)

        for k in range(n, len(axes_arr)):
            axes_arr[k].axis("off")

        handles, labels = axes_arr[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
        fig.suptitle(
            f"Trajectory diagnostics ({group_name}) | outcome={outcome_name} | "
            f"focus={focus_model} vs reference={reference_model}",
            y=0.995,
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = output_dir / f"trajectory_{group_name}_{_short_name(outcome_name)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _plot_handpicked(indices: np.ndarray) -> Optional[Path]:
        if indices.size == 0:
            return None
        idx_unique = np.unique(indices.astype(np.int64))
        idx_unique = idx_unique[(idx_unique >= 0) & (idx_unique < y_true.shape[0])]
        if idx_unique.size == 0:
            print("[trajectory] handpicked indices were empty after bounds filtering.")
            return None

        n = int(idx_unique.size)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.6 * nrows), sharex=True)
        axes_arr = np.array(axes).reshape(-1)
        include_t0 = y_last_obs is not None
        t = np.arange(0, y_true.shape[1] + 1) if include_t0 else np.arange(1, y_true.shape[1] + 1)
        focus_line = focus_model if focus_model in predictions_by_model else None

        for ax_i, win_idx in enumerate(idx_unique):
            ax = axes_arr[ax_i]
            truth = y_true[win_idx, :, outcome_idx].cpu().numpy()
            y0 = y_last_obs[win_idx, outcome_idx].detach().cpu().numpy() if include_t0 else None
            truth_plot = np.concatenate([np.array([float(y0)]), truth]) if include_t0 else truth
            ax.plot(t, truth_plot, color="#111111", linewidth=2.2, label="truth")

            ref_pred = predictions_by_model[reference_model][win_idx, :, outcome_idx].cpu().numpy()
            ref_plot = np.concatenate([np.array([float(y0)]), ref_pred]) if include_t0 else ref_pred
            ax.plot(t, ref_plot, color="#6b7280", linestyle="--", linewidth=1.9, label=reference_model)

            if focus_line is not None:
                focus_pred = predictions_by_model[focus_line][win_idx, :, outcome_idx].cpu().numpy()
                focus_plot = np.concatenate([np.array([float(y0)]), focus_pred]) if include_t0 else focus_pred
                ax.plot(t, focus_plot, color="#1f77b4", linewidth=1.9, label=focus_line)

            title_bits = [f"idx={int(win_idx)}"]
            if "country" in metadata_df.columns:
                title_bits.append(str(metadata_df.iloc[int(win_idx)]["country"]))
            if "fut_start_date" in metadata_df.columns:
                title_bits.append(str(pd.Timestamp(metadata_df.iloc[int(win_idx)]["fut_start_date"]).date()))
            ax.set_title(" | ".join(title_bits), fontsize=9)
            if include_t0:
                ax.axvline(0, color="#9ca3af", linewidth=0.8, alpha=0.6)
                ax.set_xticks([0, 7, 14, 21, 28, 35, 42])
            ax.grid(alpha=0.2)

        for k in range(n, len(axes_arr)):
            axes_arr[k].axis("off")

        handles, labels = axes_arr[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
        fig.suptitle(
            f"Hand-picked trajectories | outcome={outcome_name} | "
            f"{focus_model} vs {reference_model}",
            y=0.995,
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = output_dir / f"fig_trajectory_handpicked_{_short_name(outcome_name)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    samples_rows = []
    for group_name, idxs in [("focus_beats_reference", better_sel), ("focus_loses_to_reference", worse_sel)]:
        for win_idx in idxs.tolist():
            samples_rows.append(
                {
                    "group": group_name,
                    "window_index": int(win_idx),
                    "country": metadata_df.iloc[int(win_idx)]["country"] if "country" in metadata_df.columns else "",
                    "fut_start_date": (
                        str(pd.Timestamp(metadata_df.iloc[int(win_idx)]["fut_start_date"]).date())
                        if "fut_start_date" in metadata_df.columns
                        else ""
                    ),
                    "focus_late_rmse": float(focus_rmse[int(win_idx)]),
                    "reference_late_rmse": float(ref_rmse[int(win_idx)]),
                    "delta_focus_minus_reference": float(focus_rmse[int(win_idx)] - ref_rmse[int(win_idx)]),
                }
            )
        plot_path = _plot_group(idxs, group_name)
        if plot_path is not None:
            saved_paths.append(plot_path)

    if handpicked_indices is not None:
        p = _plot_handpicked(handpicked_indices)
        if p is not None:
            saved_paths.append(p)

    samples_df = pd.DataFrame(samples_rows)
    samples_csv = output_dir / "trajectory_window_samples.csv"
    samples_csv_outcome = output_dir / f"trajectory_window_samples_{_short_name(outcome_name)}.csv"
    samples_df.to_csv(samples_csv, index=False)
    samples_df.to_csv(samples_csv_outcome, index=False)
    saved_paths.append(samples_csv)
    saved_paths.append(samples_csv_outcome)
    return saved_paths


def _quantile_group(values: np.ndarray, labels: List[str], quantiles: List[float]) -> np.ndarray:
    vals = values.astype(np.float64)
    cuts = [float(np.quantile(vals, q)) for q in quantiles]
    groups = np.array([labels[-1]] * len(vals), dtype=object)
    if len(labels) == 2:
        groups = np.where(vals <= cuts[0], labels[0], labels[1]).astype(object)
    elif len(labels) == 3:
        groups = np.where(vals <= cuts[0], labels[0], np.where(vals <= cuts[1], labels[1], labels[2])).astype(object)
    else:
        raise ValueError("Only 2 or 3 labels are supported.")
    return groups


def build_regime_breakdown_metrics(
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    test_raw: PanelWindows,
    buckets: HorizonBuckets,
    policy_score: np.ndarray,
    outcome_idx: int,
) -> pd.DataFrame:
    y_hist = test_raw.y_hist[:, :, outcome_idx].cpu().numpy()
    n, t = y_hist.shape
    if t < 2:
        raise ValueError("Need at least 2 history points for regime breakdown.")

    past_variance = np.var(y_hist, axis=1)
    x = np.arange(t, dtype=np.float64)
    x_center = x - np.mean(x)
    denom = np.sum(x_center * x_center)
    slope = np.sum(y_hist * x_center.reshape(1, -1), axis=1) / max(denom, 1e-8)
    outcome_level = np.mean(y_hist[:, -min(7, t) :], axis=1)

    regime_groups: Dict[str, np.ndarray] = {
        "past_variance": _quantile_group(past_variance, ["low", "mid", "high"], [1 / 3, 2 / 3]),
        "past_slope": _quantile_group(slope, ["down", "flat", "up"], [1 / 3, 2 / 3]),
        "policy_change_magnitude": _quantile_group(policy_score, ["low", "mid", "high"], [1 / 3, 2 / 3]),
        "outcome_level": _quantile_group(outcome_level, ["low", "high"], [0.5]),
    }

    overall_rmse_by_model = {
        m: compute_window_segment_rmse(pred, y_true, start_1b=1, end_1b=buckets.horizon)
        for m, pred in predictions_by_model.items()
    }
    long_rmse_by_model = {
        m: compute_window_segment_rmse(pred, y_true, start_1b=buckets.long_start, end_1b=buckets.horizon)
        for m, pred in predictions_by_model.items()
    }
    late_rmse_by_model = {
        m: compute_window_segment_rmse(pred, y_true, start_1b=buckets.late_start, end_1b=buckets.horizon)
        for m, pred in predictions_by_model.items()
    }
    ref = late_rmse_by_model.get("persistence")

    rows: List[Dict[str, float | str | int]] = []
    for regime_name, group_labels in regime_groups.items():
        for group in sorted(pd.Series(group_labels).unique()):
            mask = np.where(group_labels == group)[0]
            if mask.size == 0:
                continue
            for model_name in sorted(predictions_by_model.keys()):
                late_vals = late_rmse_by_model[model_name][mask]
                row: Dict[str, float | str | int] = {
                    "regime_type": regime_name,
                    "regime_bin": str(group),
                    "model": model_name,
                    "n_windows": int(mask.size),
                    "overall_rmse": float(np.mean(overall_rmse_by_model[model_name][mask])),
                    "long_rmse": float(np.mean(long_rmse_by_model[model_name][mask])),
                    "late_rmse": float(np.mean(late_vals)),
                }
                if ref is not None:
                    ref_vals = ref[mask]
                    row["delta_late_rmse_vs_persistence"] = float(np.mean(late_vals) - np.mean(ref_vals))
                    row["late_win_rate_vs_persistence"] = float(paired_win_rate(late_vals, ref_vals))
                else:
                    row["delta_late_rmse_vs_persistence"] = float("nan")
                    row["late_win_rate_vs_persistence"] = float("nan")
                rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["regime_type", "regime_bin", "late_rmse"]).reset_index(drop=True)


def build_incidence_regime_metrics(
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    test_raw: PanelWindows,
    buckets: HorizonBuckets,
    outcome_idx: int,
    low_incidence_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Two-way regime split:
    - low_incidence: mean recent history < threshold (per million)
    - other_incidence: remaining windows
    """
    y_hist = test_raw.y_hist[:, :, outcome_idx].cpu().numpy().astype(np.float64)
    t = int(y_hist.shape[1])
    mean_recent = np.mean(y_hist[:, -min(7, t) :], axis=1)
    low_mask = mean_recent < float(low_incidence_threshold)
    groups = {
        "low_incidence": np.where(low_mask)[0],
        "other_incidence": np.where(~low_mask)[0],
    }

    overall_rmse_by_model = {
        m: compute_window_segment_rmse(pred, y_true, start_1b=1, end_1b=buckets.horizon)
        for m, pred in predictions_by_model.items()
    }
    long_rmse_by_model = {
        m: compute_window_segment_rmse(pred, y_true, start_1b=buckets.long_start, end_1b=buckets.horizon)
        for m, pred in predictions_by_model.items()
    }
    late_rmse_by_model = {
        m: compute_window_segment_rmse(pred, y_true, start_1b=buckets.late_start, end_1b=buckets.horizon)
        for m, pred in predictions_by_model.items()
    }
    ref = late_rmse_by_model.get("persistence")

    rows: List[Dict[str, float | str | int]] = []
    for regime_name, idx in groups.items():
        if idx.size == 0:
            continue
        for model_name in sorted(predictions_by_model.keys()):
            late_vals = late_rmse_by_model[model_name][idx]
            row: Dict[str, float | str | int] = {
                "regime_type": "incidence_split",
                "regime_bin": regime_name,
                "low_incidence_threshold": float(low_incidence_threshold),
                "model": model_name,
                "n_windows": int(idx.size),
                "overall_rmse": float(np.mean(overall_rmse_by_model[model_name][idx])),
                "long_rmse": float(np.mean(long_rmse_by_model[model_name][idx])),
                "late_rmse": float(np.mean(late_vals)),
            }
            if ref is not None:
                ref_vals = ref[idx]
                row["delta_late_rmse_vs_persistence"] = float(np.mean(late_vals) - np.mean(ref_vals))
                row["late_win_rate_vs_persistence"] = float(paired_win_rate(late_vals, ref_vals))
            else:
                row["delta_late_rmse_vs_persistence"] = float("nan")
                row["late_win_rate_vs_persistence"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime_bin", "late_rmse"]).reset_index(drop=True)


def _turning_points_1b(series: np.ndarray) -> np.ndarray:
    """
    Return 1-based turning-point indices (local trend reversals) in a 1D series.
    """
    if series.size < 3:
        return np.array([], dtype=np.int64)
    d = np.diff(series.astype(np.float64))
    s = np.sign(d)
    if np.all(s == 0):
        return np.array([], dtype=np.int64)

    # Fill zero-sign segments using nearest non-zero signs.
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]

    turn_idx0 = np.where(s[:-1] * s[1:] < 0)[0] + 1  # index in original series (0-based)
    return (turn_idx0 + 1).astype(np.int64)  # 1-based


def build_shape_metrics_table(
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    outcome_idx: int,
    turning_tolerance_days: int = 7,
) -> pd.DataFrame:
    """
    Shape-focused metrics:
    - first-difference RMSE
    - direction accuracy
    - turning-point precision/recall/F1 (with tolerance)
    """
    rows: List[Dict[str, float | str]] = []
    y_true_np = y_true.detach().cpu().numpy().astype(np.float64)
    y_true_main = y_true_np[:, :, outcome_idx]
    dy_true_main = np.diff(y_true_main, axis=1)

    for model_name in sorted(predictions_by_model.keys()):
        y_pred = predictions_by_model[model_name]
        y_pred_np = y_pred.detach().cpu().numpy().astype(np.float64)

        dy_pred = np.diff(y_pred_np, axis=1)
        dy_true = np.diff(y_true_np, axis=1)
        first_diff_rmse = float(np.sqrt(np.mean((dy_pred - dy_true) ** 2)))

        dy_pred_main = np.diff(y_pred_np[:, :, outcome_idx], axis=1)
        sign_match = np.sign(dy_pred_main) == np.sign(dy_true_main)
        direction_acc = float(np.mean(sign_match))

        total_pred_turn = 0
        total_true_turn = 0
        matched_pred_turn = 0
        matched_true_turn = 0
        for i in range(y_true_main.shape[0]):
            pred_turn = _turning_points_1b(y_pred_np[i, :, outcome_idx])
            true_turn = _turning_points_1b(y_true_main[i])
            total_pred_turn += int(pred_turn.size)
            total_true_turn += int(true_turn.size)
            if pred_turn.size > 0 and true_turn.size > 0:
                matched_pred_turn += int(
                    np.sum([np.any(np.abs(true_turn - p) <= turning_tolerance_days) for p in pred_turn])
                )
                matched_true_turn += int(
                    np.sum([np.any(np.abs(pred_turn - t) <= turning_tolerance_days) for t in true_turn])
                )

        turn_precision = float(matched_pred_turn / total_pred_turn) if total_pred_turn > 0 else float("nan")
        turn_recall = float(matched_true_turn / total_true_turn) if total_true_turn > 0 else float("nan")
        if np.isnan(turn_precision) or np.isnan(turn_recall) or (turn_precision + turn_recall) <= 0.0:
            turn_f1 = float("nan")
        else:
            turn_f1 = float(2.0 * turn_precision * turn_recall / (turn_precision + turn_recall))

        rows.append(
            {
                "model": model_name,
                "first_difference_rmse": first_diff_rmse,
                "direction_accuracy": direction_acc,
                "turning_point_precision": turn_precision,
                "turning_point_recall": turn_recall,
                "turning_point_f1_tolerance_days": turn_f1,
                "turning_tolerance_days": float(turning_tolerance_days),
            }
        )
    return pd.DataFrame(rows).sort_values("first_difference_rmse").reset_index(drop=True)


def _week_segment_bounds(horizon: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Choose two week-like segments for error-slope diagnostics.
    Prefer week4 and week6 for horizon >= 42.
    """
    if horizon >= 42:
        return (22, 28), (36, 42)
    if horizon >= 35:
        return (15, 21), (29, 35)
    if horizon >= 28:
        return (8, 14), (22, 28)
    return (1, min(7, horizon)), (max(1, horizon - 6), horizon)


def build_stability_metrics_table(
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    policy_score: np.ndarray,
    outcome_idx: int,
    constant_policy_quantile: float = 0.25,
) -> pd.DataFrame:
    """
    Stability diagnostics:
    - variance ratio over rollout
    - near-constant-policy drift proxy
    - nonnegativity violation rate/magnitude
    - week-segment error slope
    """
    rows: List[Dict[str, float | str]] = []
    y_true_np = y_true.detach().cpu().numpy().astype(np.float64)
    true_var = float(np.var(y_true_np[:, :, outcome_idx]))
    true_var = max(true_var, 1e-8)
    horizon = int(y_true.shape[1])
    seg_a, seg_b = _week_segment_bounds(horizon)
    const_cut = float(np.quantile(policy_score.astype(np.float64), float(np.clip(constant_policy_quantile, 0.0, 1.0))))
    const_mask = policy_score <= const_cut
    const_n = int(np.sum(const_mask))

    for model_name in sorted(predictions_by_model.keys()):
        y_pred = predictions_by_model[model_name]
        y_pred_np = y_pred.detach().cpu().numpy().astype(np.float64)

        variance_ratio = float(np.var(y_pred_np[:, :, outcome_idx]) / true_var)

        if const_n > 0:
            pred_const = y_pred_np[const_mask, :, outcome_idx]
            constant_policy_drift_abs = float(np.mean(np.abs(pred_const[:, -1] - pred_const[:, 0])))
            constant_policy_drift_signed = float(np.mean(pred_const[:, -1] - pred_const[:, 0]))
        else:
            constant_policy_drift_abs = float("nan")
            constant_policy_drift_signed = float("nan")

        neg_mask = y_pred_np < 0.0
        nonneg_violation_rate = float(np.mean(neg_mask))
        if np.any(neg_mask):
            nonneg_violation_magnitude = float(np.mean(-y_pred_np[neg_mask]))
        else:
            nonneg_violation_magnitude = 0.0

        rmse_seg_a = float(
            np.mean(
                compute_window_segment_rmse(
                    y_pred=y_pred,
                    y_true=y_true,
                    start_1b=seg_a[0],
                    end_1b=seg_a[1],
                )
            )
        )
        rmse_seg_b = float(
            np.mean(
                compute_window_segment_rmse(
                    y_pred=y_pred,
                    y_true=y_true,
                    start_1b=seg_b[0],
                    end_1b=seg_b[1],
                )
            )
        )
        denom = float(max(1, (seg_b[0] - seg_a[0]) / 7.0))
        week_error_slope = float((rmse_seg_b - rmse_seg_a) / denom)

        rows.append(
            {
                "model": model_name,
                "variance_ratio_rollout": variance_ratio,
                "constant_policy_quantile": float(constant_policy_quantile),
                "constant_policy_n_windows": float(const_n),
                "constant_policy_drift_abs": constant_policy_drift_abs,
                "constant_policy_drift_signed": constant_policy_drift_signed,
                "nonneg_violation_rate": nonneg_violation_rate,
                "nonneg_violation_magnitude": nonneg_violation_magnitude,
                "error_rmse_segment_a_start": float(seg_a[0]),
                "error_rmse_segment_a_end": float(seg_a[1]),
                "error_rmse_segment_b_start": float(seg_b[0]),
                "error_rmse_segment_b_end": float(seg_b[1]),
                "rmse_segment_a": rmse_seg_a,
                "rmse_segment_b": rmse_seg_b,
                "week_error_slope": week_error_slope,
            }
        )
    return pd.DataFrame(rows).sort_values("week_error_slope").reset_index(drop=True)


def _first_crossing_day(series: np.ndarray, threshold: float) -> float:
    idx = np.where(series >= threshold)[0]
    if idx.size == 0:
        return float("nan")
    return float(idx[0] + 1)


def _safe_event_f1(y_true_event: np.ndarray, y_pred_event: np.ndarray) -> float:
    y_true_event = y_true_event.astype(bool)
    y_pred_event = y_pred_event.astype(bool)
    tp = int(np.sum(y_true_event & y_pred_event))
    fp = int(np.sum(~y_true_event & y_pred_event))
    fn = int(np.sum(y_true_event & ~y_pred_event))
    denom = float(2 * tp + fp + fn)
    if denom <= 0.0:
        return float("nan")
    return float(2.0 * tp / denom)


def _safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    if y_true.size == 0 or np.all(y_true == y_true[0]):
        return float("nan")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = float(np.sum(pos[:, None] > neg[None, :]))
    ties = float(np.sum(np.isclose(pos[:, None], neg[None, :], rtol=1e-10, atol=1e-12)))
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def _safe_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    order = np.argsort(-y_score)
    k_eff = int(max(1, min(k, y_true.size)))
    gains = np.maximum(y_true[order[:k_eff]], 0.0)
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))

    ideal_order = np.argsort(-y_true)
    ideal_gains = np.maximum(y_true[ideal_order[:k_eff]], 0.0)
    idcg = float(np.sum(ideal_gains * discounts))
    if idcg <= 1e-12:
        return float("nan")
    return float(dcg / idcg)


def _ranking_metrics_by_date(
    metadata_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    horizon_1b: int,
    positive_quantile: float,
) -> Tuple[float, float, int]:
    """
    Compute date-wise country risk ranking metrics at a specific horizon.
    """
    if "fut_start_date" not in metadata_df.columns or horizon_1b > y_true.shape[1]:
        return float("nan"), float("nan"), 0

    temp = pd.DataFrame(
        {
            "fut_start_date": pd.to_datetime(metadata_df["fut_start_date"]),
            "pred": y_pred[:, horizon_1b - 1],
            "true": y_true[:, horizon_1b - 1],
        }
    )
    auc_vals: List[float] = []
    ndcg_vals: List[float] = []
    groups = 0
    q = float(np.clip(positive_quantile, 0.0, 1.0))
    for _, g in temp.groupby("fut_start_date"):
        if g.shape[0] < 5:
            continue
        y_true_g = g["true"].to_numpy(dtype=np.float64)
        y_pred_g = g["pred"].to_numpy(dtype=np.float64)
        cut = float(np.quantile(y_true_g, q))
        y_bin = (y_true_g >= cut).astype(np.int64)
        auc = _safe_binary_auc(y_bin, y_pred_g)
        ndcg = _safe_ndcg(y_true_g, y_pred_g, k=min(10, g.shape[0]))
        if not np.isnan(auc):
            auc_vals.append(float(auc))
        if not np.isnan(ndcg):
            ndcg_vals.append(float(ndcg))
        groups += 1

    auc_mean = float(np.mean(auc_vals)) if auc_vals else float("nan")
    ndcg_mean = float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")
    return auc_mean, ndcg_mean, int(groups)


def build_utility_metrics_table(
    predictions_by_model: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    y_last_obs: torch.Tensor,
    metadata_df: pd.DataFrame,
    outcome_idx: int,
    threshold_value: float = 50.0,
    risk_positive_quantile: float = 0.75,
    actionable_rise_delta: float = 10.0,
) -> pd.DataFrame:
    """
    Planning utility metrics:
    - threshold crossing timing/F1
    - country risk ranking quality at h28 and h42
    - actionable lead time for rise warnings
    """
    rows: List[Dict[str, float | str]] = []
    y_true_np = y_true.detach().cpu().numpy().astype(np.float64)
    y_true_main = y_true_np[:, :, outcome_idx]
    y_last = y_last_obs.detach().cpu().numpy().astype(np.float64)[:, outcome_idx]
    horizon = int(y_true.shape[1])

    true_cross_day = np.array([_first_crossing_day(row, threshold=float(threshold_value)) for row in y_true_main], dtype=np.float64)
    true_event = ~np.isnan(true_cross_day)

    true_rise_day = np.array([_first_crossing_day(row, threshold=float(y_last[i] + actionable_rise_delta)) for i, row in enumerate(y_true_main)], dtype=np.float64)
    true_rise_event = ~np.isnan(true_rise_day)

    h_eval_28 = 28 if horizon >= 28 else horizon
    h_eval_42 = 42 if horizon >= 42 else horizon

    for model_name in sorted(predictions_by_model.keys()):
        y_pred_main = predictions_by_model[model_name].detach().cpu().numpy().astype(np.float64)[:, :, outcome_idx]

        pred_cross_day = np.array(
            [_first_crossing_day(row, threshold=float(threshold_value)) for row in y_pred_main],
            dtype=np.float64,
        )
        pred_event = ~np.isnan(pred_cross_day)
        crossing_f1 = _safe_event_f1(true_event, pred_event)
        cross_timing_mask = true_event & pred_event
        crossing_timing_mae = (
            float(np.mean(np.abs(pred_cross_day[cross_timing_mask] - true_cross_day[cross_timing_mask])))
            if np.any(cross_timing_mask)
            else float("nan")
        )

        pred_rise_day = np.array(
            [_first_crossing_day(row, threshold=float(y_last[i] + actionable_rise_delta)) for i, row in enumerate(y_pred_main)],
            dtype=np.float64,
        )
        pred_rise_event = ~np.isnan(pred_rise_day)
        rise_detection_f1 = _safe_event_f1(true_rise_event, pred_rise_event)

        actionable_mask = true_rise_event & pred_rise_event & (pred_rise_day <= true_rise_day)
        actionable_lead = (
            float(np.mean(true_rise_day[actionable_mask] - pred_rise_day[actionable_mask]))
            if np.any(actionable_mask)
            else float("nan")
        )
        actionable_recall = float(np.mean(actionable_mask[true_rise_event])) if np.any(true_rise_event) else float("nan")

        auc_h28, ndcg_h28, ranking_groups_h28 = _ranking_metrics_by_date(
            metadata_df=metadata_df,
            y_pred=y_pred_main,
            y_true=y_true_main,
            horizon_1b=h_eval_28,
            positive_quantile=risk_positive_quantile,
        )
        auc_h42, ndcg_h42, ranking_groups_h42 = _ranking_metrics_by_date(
            metadata_df=metadata_df,
            y_pred=y_pred_main,
            y_true=y_true_main,
            horizon_1b=h_eval_42,
            positive_quantile=risk_positive_quantile,
        )

        rows.append(
            {
                "model": model_name,
                "threshold_value": float(threshold_value),
                "threshold_crossing_f1": crossing_f1,
                "threshold_crossing_timing_mae_days": crossing_timing_mae,
                "risk_positive_quantile": float(risk_positive_quantile),
                f"risk_auc_h{h_eval_28}": auc_h28,
                f"risk_ndcg_h{h_eval_28}": ndcg_h28,
                f"risk_auc_h{h_eval_42}": auc_h42,
                f"risk_ndcg_h{h_eval_42}": ndcg_h42,
                "risk_ranking_groups_h28": float(ranking_groups_h28),
                "risk_ranking_groups_h42": float(ranking_groups_h42),
                "actionable_rise_delta": float(actionable_rise_delta),
                "rise_detection_f1": rise_detection_f1,
                "actionable_lead_time_days": actionable_lead,
                "actionable_recall": actionable_recall,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold_crossing_timing_mae_days").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended Oxford evaluation with short/long tables")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: label=/path/to/checkpoint.pt ; repeat for multiple models",
    )
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/oxford_extended")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--allow_horizon_truncate",
        action="store_true",
        help=(
            "Allow evaluating checkpoints whose config.forecast_horizon differs from the test horizon. "
            "Useful for evaluating a model trained with longer horizon on shorter-horizon windows."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--short_horizon_end", type=int, default=None)
    parser.add_argument("--long_horizon_start", type=int, default=None)
    parser.add_argument("--late_horizon_start", type=int, default=None)
    parser.add_argument("--peak_outcome", type=str, default="new_cases_smoothed_per_million")
    parser.add_argument("--policy_change_method", type=str, choices=["quantile", "threshold"], default="quantile")
    parser.add_argument("--policy_change_quantile", type=float, default=0.75)
    parser.add_argument("--policy_change_threshold", type=float, default=0.0)
    parser.add_argument(
        "--clip_nonnegative",
        action="store_true",
        help="Clamp model/baseline predictions to >= 0 before metric computation.",
    )
    parser.add_argument(
        "--pred_cache_dir",
        type=str,
        default=None,
        help="Optional directory for cached raw predictions; defaults to <output_dir>/pred_cache",
    )
    parser.add_argument("--recompute_predictions", action="store_true", help="Ignore cached predictions")
    parser.add_argument(
        "--drop_negligible_countries",
        action="store_true",
        help="Drop countries entirely when their recent-incidence never exceeds threshold.",
    )
    parser.add_argument(
        "--negligible_outcome",
        type=str,
        default="new_cases_smoothed_per_million",
        help="Outcome used to determine negligible epidemic dynamics.",
    )
    parser.add_argument(
        "--negligible_history_len",
        type=int,
        default=21,
        help="History length for country-level recent incidence mean in negligible-country filtering.",
    )
    parser.add_argument(
        "--negligible_threshold",
        type=float,
        default=5.0,
        help="Country is dropped if max recent mean incidence stays below this threshold.",
    )
    parser.add_argument(
        "--no_baselines",
        action="store_true",
        help=(
            "Skip all built-in baselines "
            "(mean, persistence, seasonal_naive_7d, linear_trend, last_7day_mean, AR/ARX/boosted lag, LSTM seq2seq)"
        ),
    )
    parser.add_argument("--include_ml_baselines", action="store_true", help="Enable AR/ARX ridge and boosted lag baselines.")
    parser.add_argument("--ml_lag_len", type=int, default=21, help="Lag length for AR/ARX/boosted baselines.")
    parser.add_argument("--ml_ridge_alpha", type=float, default=1.0, help="Ridge regularization alpha for AR/ARX baselines.")
    parser.add_argument(
        "--ml_boost_backend",
        type=str,
        default="hist_gbm",
        choices=["auto", "xgboost", "lightgbm", "hist_gbm", "none"],
        help="Boosting backend preference for lag baseline.",
    )
    parser.add_argument("--ml_boost_max_iter", type=int, default=120, help="Boosting iterations/estimators for lag baseline.")
    parser.add_argument("--ml_max_train_windows", type=int, default=6000, help="Max train windows used by boosted lag baseline.")
    parser.add_argument("--ml_skip_boosted", action="store_true", help="Skip boosted lag baseline (use AR/ARX only).")
    parser.add_argument(
        "--ml_include_country_context",
        action="store_true",
        help="Append country one-hot features to AR/ARX/boosted lag baselines.",
    )
    parser.add_argument("--include_lstm_baseline", action="store_true", help="Enable trained seq2seq LSTM baseline.")
    parser.add_argument("--lstm_hidden_dim", type=int, default=96, help="Hidden size for LSTM baseline.")
    parser.add_argument("--lstm_epochs", type=int, default=12, help="Training epochs for LSTM baseline.")
    parser.add_argument("--lstm_batch_size", type=int, default=128, help="Mini-batch size for LSTM baseline.")
    parser.add_argument("--lstm_lr", type=float, default=1e-3, help="Learning rate for LSTM baseline.")
    parser.add_argument("--lstm_max_train_windows", type=int, default=12000, help="Max train windows for LSTM baseline fitting.")
    parser.add_argument("--lstm_val_fraction", type=float, default=0.1, help="Validation fraction for LSTM early stopping.")
    parser.add_argument("--lstm_patience", type=int, default=4, help="Early-stopping patience (epochs) for LSTM baseline.")
    parser.add_argument("--lstm_tf_start", type=float, default=0.9, help="Teacher-forcing ratio at epoch 1 for LSTM baseline.")
    parser.add_argument("--lstm_tf_end", type=float, default=0.2, help="Teacher-forcing ratio at final epoch for LSTM baseline.")
    parser.add_argument("--lstm_include_country_context", action="store_true", help="Append country one-hot context in LSTM baseline.")
    parser.add_argument(
        "--lstm_no_future_policy",
        action="store_true",
        help="Disable future policy sequence input for LSTM baseline decoder.",
    )
    parser.add_argument(
        "--save_trajectory_plots",
        action="store_true",
        help="Save trajectory diagnostics for windows where a focus model beats/loses to a reference model.",
    )
    parser.add_argument(
        "--trajectory_focus_model",
        type=str,
        default="planning_stress",
        help="Model label used to define beat/lose trajectory groups.",
    )
    parser.add_argument(
        "--trajectory_reference_model",
        type=str,
        default="persistence",
        help="Reference model label for beat/lose trajectory groups.",
    )
    parser.add_argument(
        "--trajectory_samples_per_group",
        type=int,
        default=20,
        help="Number of windows to sample for each trajectory group.",
    )
    parser.add_argument(
        "--trajectory_outcome",
        type=str,
        default="new_cases_smoothed_per_million",
        help="Outcome column to plot in trajectory diagnostics.",
    )
    parser.add_argument(
        "--trajectory_window_indices",
        type=str,
        default="",
        help="Optional comma-separated 0-based window indices for an explicit hand-picked trajectory figure.",
    )
    parser.add_argument(
        "--save_regime_breakdown",
        action="store_true",
        help="Save grouped regime breakdown metrics (variance/slope/policy-change/outcome-level).",
    )
    parser.add_argument(
        "--regime_outcome",
        type=str,
        default="new_cases_smoothed_per_million",
        help="Outcome column used for regime feature construction.",
    )
    parser.add_argument(
        "--save_incidence_regime_metrics",
        action="store_true",
        help="Save explicit low-incidence vs other-incidence comparison table.",
    )
    parser.add_argument(
        "--incidence_outcome",
        type=str,
        default="new_cases_smoothed_per_million",
        help="Outcome column used to define low-incidence windows.",
    )
    parser.add_argument(
        "--low_incidence_threshold",
        type=float,
        default=5.0,
        help="Low-incidence threshold on mean recent history (per million).",
    )
    parser.add_argument("--save_shape_metrics", action="store_true", help="Save shape-focused diagnostics table.")
    parser.add_argument("--turning_tolerance_days", type=int, default=7, help="Tolerance window for turning-point F1.")
    parser.add_argument("--save_stability_metrics", action="store_true", help="Save long-horizon stability diagnostics table.")
    parser.add_argument(
        "--constant_policy_quantile",
        type=float,
        default=0.25,
        help="Quantile cutoff for near-constant policy windows in drift diagnostics.",
    )
    parser.add_argument("--save_utility_metrics", action="store_true", help="Save planning utility diagnostics table.")
    parser.add_argument(
        "--utility_outcome",
        type=str,
        default="new_cases_smoothed_per_million",
        help="Outcome column used for utility diagnostics.",
    )
    parser.add_argument(
        "--utility_threshold",
        type=float,
        default=50.0,
        help="Threshold for crossing metrics (in original outcome units).",
    )
    parser.add_argument(
        "--risk_positive_quantile",
        type=float,
        default=0.75,
        help="Positive class quantile for per-date risk AUC ranking metric.",
    )
    parser.add_argument(
        "--actionable_rise_delta",
        type=float,
        default=10.0,
        help="Absolute rise above current level used for actionable lead-time metric.",
    )
    args = parser.parse_args()

    model_specs = [parse_model_spec(s) for s in args.model]
    model_labels = [m[0] for m in model_specs]
    if len(set(model_labels)) != len(model_labels):
        raise ValueError(f"Duplicate model labels found in --model arguments: {model_labels}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_cache_dir = Path(args.pred_cache_dir) if args.pred_cache_dir else out_dir / "pred_cache"

    # Load first checkpoint for shared column names + country mapping
    _, ref_ckpt = model_specs[0]
    _, _, _, ref_country_to_idx, policy_cols, outcome_cols, state_cols = load_checkpoint(ref_ckpt, device=args.device)
    if not policy_cols or not outcome_cols:
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
        country_to_idx=ref_country_to_idx,
    )
    train_raw, test_raw = subset_windows(train_raw, test_raw, max_windows=args.max_windows, seed=args.seed)
    country_filter_meta: Optional[dict] = None
    country_exclusion_df: Optional[pd.DataFrame] = None
    excluded_countries: List[str] = []
    if args.drop_negligible_countries:
        train_before, test_before = int(len(train_raw)), int(len(test_raw))
        train_raw, test_raw, country_exclusion_df, excluded_countries = filter_negligible_countries(
            train_raw=train_raw,
            test_raw=test_raw,
            outcome_cols=outcome_cols,
            outcome_name=args.negligible_outcome,
            history_len=int(max(1, args.negligible_history_len)),
            threshold=float(args.negligible_threshold),
        )
        country_filter_meta = {
            "enabled": True,
            "method": "drop_country_if_max_recent_mean_below_threshold",
            "outcome": str(args.negligible_outcome),
            "history_len": int(max(1, args.negligible_history_len)),
            "threshold": float(args.negligible_threshold),
            "countries_dropped_n": int(len(excluded_countries)),
            "countries_dropped": excluded_countries,
            "train_windows_before": int(train_before),
            "train_windows_after": int(len(train_raw)),
            "test_windows_before": int(test_before),
            "test_windows_after": int(len(test_raw)),
        }
        print(
            "Country filter enabled: "
            f"dropped {len(excluded_countries)} countries, "
            f"test windows {test_before}->{len(test_raw)}, "
            f"train windows {train_before}->{len(train_raw)}"
        )
    else:
        country_filter_meta = {"enabled": False}

    horizon = int(test_raw.y_fut.shape[1])
    if args.short_horizon_end is None or args.long_horizon_start is None or args.late_horizon_start is None:
        defaults = default_horizon_buckets(horizon)
        short_end = defaults.short_end if args.short_horizon_end is None else args.short_horizon_end
        long_start = defaults.long_start if args.long_horizon_start is None else args.long_horizon_start
        late_start = defaults.late_start if args.late_horizon_start is None else args.late_horizon_start
    else:
        short_end = args.short_horizon_end
        long_start = args.long_horizon_start
        late_start = args.late_horizon_start
    buckets = HorizonBuckets(
        short_end=int(short_end),
        long_start=int(long_start),
        late_start=int(late_start),
        horizon=horizon,
    )
    buckets.validate()

    if args.peak_outcome not in outcome_cols:
        raise ValueError(f"--peak_outcome '{args.peak_outcome}' not found in outcomes: {outcome_cols}")
    peak_outcome_idx = outcome_cols.index(args.peak_outcome)
    if args.trajectory_outcome not in outcome_cols:
        raise ValueError(f"--trajectory_outcome '{args.trajectory_outcome}' not found in outcomes: {outcome_cols}")
    if args.regime_outcome not in outcome_cols:
        raise ValueError(f"--regime_outcome '{args.regime_outcome}' not found in outcomes: {outcome_cols}")
    if args.incidence_outcome not in outcome_cols:
        raise ValueError(f"--incidence_outcome '{args.incidence_outcome}' not found in outcomes: {outcome_cols}")
    if args.utility_outcome not in outcome_cols:
        raise ValueError(f"--utility_outcome '{args.utility_outcome}' not found in outcomes: {outcome_cols}")
    trajectory_outcome_idx = outcome_cols.index(args.trajectory_outcome)
    regime_outcome_idx = outcome_cols.index(args.regime_outcome)
    incidence_outcome_idx = outcome_cols.index(args.incidence_outcome)
    utility_outcome_idx = outcome_cols.index(args.utility_outcome)

    y_true = test_raw.y_fut.to(torch.float32)
    policy_score, policy_mask = compute_policy_change_mask(
        a_fut=test_raw.a_fut,
        method=args.policy_change_method,
        quantile=args.policy_change_quantile,
        threshold=args.policy_change_threshold,
    )
    policy_subset_n = int(policy_mask.sum())
    policy_subset_ratio = float(policy_subset_n / max(len(policy_mask), 1))

    print(f"Evaluating on {len(test_raw)} windows, horizon={horizon}")
    print(
        f"Short: 1..{buckets.short_end} | Long: {buckets.long_start}..{buckets.horizon} | "
        f"Late headline: {buckets.late_start}..{buckets.horizon}"
    )
    print(
        f"Policy-change subset: n={policy_subset_n}/{len(policy_mask)} "
        f"({policy_subset_ratio * 100:.1f}%) method={args.policy_change_method}"
    )
    if args.clip_nonnegative:
        print("Prediction post-processing: clip_nonnegative=True (all predictions clamped to >= 0)")

    metrics_rows: List[Dict[str, float | str]] = []
    predictions_by_model: Dict[str, torch.Tensor] = {}

    def _clip_if_needed(y_pred: torch.Tensor) -> torch.Tensor:
        return torch.clamp(y_pred, min=0.0) if args.clip_nonnegative else y_pred

    if not args.no_baselines:
        y_pred_mean = _clip_if_needed(compute_mean_baseline_prediction(train_raw, test_raw))
        predictions_by_model["mean"] = y_pred_mean
        metrics_rows.append(
            summarize_model_metrics(
                model_name="mean",
                y_pred=y_pred_mean,
                y_true=y_true,
                a_fut_raw=test_raw.a_fut,
                buckets=buckets,
                policy_subset_mask=policy_mask,
                peak_outcome_idx=peak_outcome_idx,
            )
        )

        y_pred_persist = _clip_if_needed(compute_persistence_prediction(test_raw))
        predictions_by_model["persistence"] = y_pred_persist
        metrics_rows.append(
            summarize_model_metrics(
                model_name="persistence",
                y_pred=y_pred_persist,
                y_true=y_true,
                a_fut_raw=test_raw.a_fut,
                buckets=buckets,
                policy_subset_mask=policy_mask,
                peak_outcome_idx=peak_outcome_idx,
            )
        )

        y_pred_seasonal = _clip_if_needed(compute_seasonal_naive_prediction(test_raw, seasonal_period=7))
        predictions_by_model["seasonal_naive_7d"] = y_pred_seasonal
        metrics_rows.append(
            summarize_model_metrics(
                model_name="seasonal_naive_7d",
                y_pred=y_pred_seasonal,
                y_true=y_true,
                a_fut_raw=test_raw.a_fut,
                buckets=buckets,
                policy_subset_mask=policy_mask,
                peak_outcome_idx=peak_outcome_idx,
            )
        )

        y_pred_linear_trend = _clip_if_needed(compute_linear_trend_prediction(test_raw))
        predictions_by_model["linear_trend"] = y_pred_linear_trend
        metrics_rows.append(
            summarize_model_metrics(
                model_name="linear_trend",
                y_pred=y_pred_linear_trend,
                y_true=y_true,
                a_fut_raw=test_raw.a_fut,
                buckets=buckets,
                policy_subset_mask=policy_mask,
                peak_outcome_idx=peak_outcome_idx,
            )
        )

        y_pred_last7 = _clip_if_needed(compute_last_k_mean_prediction(test_raw=test_raw, k=7))
        predictions_by_model["last_7day_mean"] = y_pred_last7
        metrics_rows.append(
            summarize_model_metrics(
                model_name="last_7day_mean",
                y_pred=y_pred_last7,
                y_true=y_true,
                a_fut_raw=test_raw.a_fut,
                buckets=buckets,
                policy_subset_mask=policy_mask,
                peak_outcome_idx=peak_outcome_idx,
            )
        )

        if args.include_ml_baselines:
            print(
                f"Evaluating AR/ARX/boosted lag baselines "
                f"(lag={args.ml_lag_len}, ridge_alpha={args.ml_ridge_alpha}, "
                f"country_context={bool(args.ml_include_country_context)})"
            )
            y_pred_ar = _clip_if_needed(
                compute_ar_ridge_prediction(
                    train_raw=train_raw,
                    test_raw=test_raw,
                    lag_len=args.ml_lag_len,
                    alpha=args.ml_ridge_alpha,
                    include_country_context=bool(args.ml_include_country_context),
                )
            )
            predictions_by_model["ar_ridge_lag"] = y_pred_ar
            metrics_rows.append(
                summarize_model_metrics(
                    model_name="ar_ridge_lag",
                    y_pred=y_pred_ar,
                    y_true=y_true,
                    a_fut_raw=test_raw.a_fut,
                    buckets=buckets,
                    policy_subset_mask=policy_mask,
                    peak_outcome_idx=peak_outcome_idx,
                )
            )

            y_pred_arx = _clip_if_needed(
                compute_arx_ridge_prediction(
                    train_raw=train_raw,
                    test_raw=test_raw,
                    lag_len=args.ml_lag_len,
                    alpha=args.ml_ridge_alpha,
                    include_country_context=bool(args.ml_include_country_context),
                )
            )
            predictions_by_model["arx_ridge_lag_policy"] = y_pred_arx
            metrics_rows.append(
                summarize_model_metrics(
                    model_name="arx_ridge_lag_policy",
                    y_pred=y_pred_arx,
                    y_true=y_true,
                    a_fut_raw=test_raw.a_fut,
                    buckets=buckets,
                    policy_subset_mask=policy_mask,
                    peak_outcome_idx=peak_outcome_idx,
                )
            )

            if args.ml_skip_boosted:
                print("Skipping boosted lag baseline by flag: --ml_skip_boosted")
            else:
                y_pred_boost, boost_backend = compute_boosted_lag_prediction(
                    train_raw=train_raw,
                    test_raw=test_raw,
                    lag_len=args.ml_lag_len,
                    max_train_windows=args.ml_max_train_windows,
                    include_future_policy=True,
                    include_country_context=bool(args.ml_include_country_context),
                    backend=args.ml_boost_backend,
                    max_iter=args.ml_boost_max_iter,
                    random_state=args.seed,
                )
                if y_pred_boost is None:
                    print(f"Skipping boosted lag baseline: {boost_backend}")
                else:
                    print(f"Boosted lag baseline backend: {boost_backend}")
                    y_pred_boost = _clip_if_needed(y_pred_boost)
                    predictions_by_model["boosted_lag_policy"] = y_pred_boost
                    metrics_rows.append(
                        summarize_model_metrics(
                            model_name="boosted_lag_policy",
                            y_pred=y_pred_boost,
                            y_true=y_true,
                            a_fut_raw=test_raw.a_fut,
                            buckets=buckets,
                            policy_subset_mask=policy_mask,
                            peak_outcome_idx=peak_outcome_idx,
                        )
                    )

        if args.include_lstm_baseline:
            print(
                f"Evaluating LSTM seq2seq baseline "
                f"(hidden={args.lstm_hidden_dim}, epochs={args.lstm_epochs}, "
                f"country_context={bool(args.lstm_include_country_context)}, "
                f"future_policy={not bool(args.lstm_no_future_policy)})"
            )
            y_pred_lstm = _clip_if_needed(
                compute_lstm_seq2seq_prediction(
                    train_raw=train_raw,
                    test_raw=test_raw,
                    include_future_policy=(not bool(args.lstm_no_future_policy)),
                    include_country_context=bool(args.lstm_include_country_context),
                    hidden_dim=int(args.lstm_hidden_dim),
                    epochs=int(args.lstm_epochs),
                    batch_size=int(args.lstm_batch_size),
                    lr=float(args.lstm_lr),
                    max_train_windows=int(args.lstm_max_train_windows),
                    val_fraction=float(args.lstm_val_fraction),
                    patience=int(args.lstm_patience),
                    teacher_forcing_start=float(args.lstm_tf_start),
                    teacher_forcing_end=float(args.lstm_tf_end),
                    seed=int(args.seed),
                    device=args.device,
                )
            )
            predictions_by_model["lstm_seq2seq"] = y_pred_lstm
            metrics_rows.append(
                summarize_model_metrics(
                    model_name="lstm_seq2seq",
                    y_pred=y_pred_lstm,
                    y_true=y_true,
                    a_fut_raw=test_raw.a_fut,
                    buckets=buckets,
                    policy_subset_mask=policy_mask,
                    peak_outcome_idx=peak_outcome_idx,
                )
            )

    for label, ckpt_path in model_specs:
        print(f"Evaluating model '{label}' from {ckpt_path}")
        y_pred = predict_crt_from_checkpoint(
            model_label=label,
            checkpoint_path=ckpt_path,
            test_raw=test_raw,
            device=args.device,
            log1p_outcomes=log1p_outcomes,
            reference_country_to_idx=ref_country_to_idx,
            cache_dir=pred_cache_dir,
            use_cache=(not args.recompute_predictions),
            batch_size=args.batch_size,
            allow_horizon_truncate=bool(args.allow_horizon_truncate),
        )
        y_pred = _clip_if_needed(y_pred)
        predictions_by_model[label] = y_pred
        metrics_rows.append(
            summarize_model_metrics(
                model_name=label,
                y_pred=y_pred,
                y_true=y_true,
                a_fut_raw=test_raw.a_fut,
                buckets=buckets,
                policy_subset_mask=policy_mask,
                peak_outcome_idx=peak_outcome_idx,
            )
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values("model").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "all_metrics_full.csv", index=False)

    short_df = build_short_term_table(metrics_df, buckets)
    long_df = build_long_term_table(metrics_df, buckets)

    short_metric_cols = [c for c in short_df.columns if c != "model"]
    long_metric_cols = [c for c in long_df.columns if c not in {"model", "policy_subset_n"}]

    save_table_outputs(
        table_df=short_df,
        csv_path=out_dir / "short_term_metrics.csv",
        md_path=out_dir / "short_term_metrics.md",
        metric_columns=short_metric_cols,
    )
    save_table_outputs(
        table_df=long_df,
        csv_path=out_dir / "long_term_metrics.csv",
        md_path=out_dir / "long_term_metrics.md",
        metric_columns=long_metric_cols,
    )

    persistence_cmp_df = None
    if "persistence" in predictions_by_model:
        persistence_cmp_df = build_persistence_comparison_table(
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            policy_subset_mask=policy_mask,
            buckets=buckets,
            reference_model="persistence",
        )
        persistence_cmp_csv = out_dir / "planning_vs_persistence.csv"
        persistence_cmp_md = out_dir / "planning_vs_persistence.md"
        persistence_cmp_df.to_csv(persistence_cmp_csv, index=False)
        write_plain_markdown_table(persistence_cmp_df, persistence_cmp_md, precision=4)

    horizon_profile_path = out_dir / "fig_horizon_profile.png"
    horizon_profile_ok = save_horizon_profile_plot(metrics_df=metrics_df, out_path=horizon_profile_path)

    main_table_df = build_main_results_table(
        metrics_df=metrics_df,
        short_df=short_df,
        long_df=long_df,
        persistence_cmp_df=persistence_cmp_df,
    )
    main_table_csv = out_dir / "table_main_results.csv"
    main_table_md = out_dir / "table_main_results.md"
    main_table_df.to_csv(main_table_csv, index=False)
    write_plain_markdown_table(main_table_df, main_table_md, precision=4)

    policy_subset_table_df = None
    policy_subset_table_csv = out_dir / "table_policy_change_subset.csv"
    policy_subset_table_md = out_dir / "table_policy_change_subset.md"
    if persistence_cmp_df is not None:
        policy_subset_table_df = persistence_cmp_df.copy().sort_values(
            "delta_policy_subset_late_rmse_vs_persistence"
        ).reset_index(drop=True)
        policy_subset_table_df.to_csv(policy_subset_table_csv, index=False)
        write_plain_markdown_table(policy_subset_table_df, policy_subset_table_md, precision=4)

    trajectory_paths: List[Path] = []
    if args.save_trajectory_plots:
        handpicked_idx = None
        if args.trajectory_window_indices.strip():
            try:
                vals = [int(v.strip()) for v in args.trajectory_window_indices.split(",") if v.strip()]
            except ValueError as e:
                raise ValueError(
                    f"Invalid --trajectory_window_indices '{args.trajectory_window_indices}'. "
                    "Use comma-separated integers, e.g. 12,57,103,241"
                ) from e
            handpicked_idx = np.array(vals, dtype=np.int64)
        trajectory_paths = save_trajectory_diagnostics(
            output_dir=out_dir,
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            y_last_obs=test_raw.y_hist[:, -1, :],
            metadata_df=test_raw.metadata,
            buckets=buckets,
            focus_model=args.trajectory_focus_model,
            reference_model=args.trajectory_reference_model,
            outcome_idx=trajectory_outcome_idx,
            outcome_name=args.trajectory_outcome,
            samples_per_group=args.trajectory_samples_per_group,
            seed=args.seed,
            handpicked_indices=handpicked_idx,
        )

    regime_df = None
    if args.save_regime_breakdown:
        regime_df = build_regime_breakdown_metrics(
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            test_raw=test_raw,
            buckets=buckets,
            policy_score=policy_score,
            outcome_idx=regime_outcome_idx,
        )
        regime_csv = out_dir / "regime_breakdown_metrics.csv"
        regime_md = out_dir / "regime_breakdown_metrics.md"
        regime_df.to_csv(regime_csv, index=False)
        write_plain_markdown_table(regime_df, regime_md, precision=4)
        regime_thesis_csv = out_dir / "table_regime_breakdown.csv"
        regime_thesis_md = out_dir / "table_regime_breakdown.md"
        regime_df.to_csv(regime_thesis_csv, index=False)
        write_plain_markdown_table(regime_df, regime_thesis_md, precision=4)

    incidence_df = None
    if args.save_incidence_regime_metrics:
        incidence_df = build_incidence_regime_metrics(
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            test_raw=test_raw,
            buckets=buckets,
            outcome_idx=incidence_outcome_idx,
            low_incidence_threshold=float(args.low_incidence_threshold),
        )
        incidence_csv = out_dir / "incidence_regime_metrics.csv"
        incidence_md = out_dir / "incidence_regime_metrics.md"
        incidence_df.to_csv(incidence_csv, index=False)
        write_plain_markdown_table(incidence_df, incidence_md, precision=4)

    shape_df = None
    if args.save_shape_metrics:
        shape_df = build_shape_metrics_table(
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            outcome_idx=utility_outcome_idx,
            turning_tolerance_days=int(max(1, args.turning_tolerance_days)),
        )
        shape_csv = out_dir / "shape_metrics.csv"
        shape_md = out_dir / "shape_metrics.md"
        shape_df.to_csv(shape_csv, index=False)
        write_plain_markdown_table(shape_df, shape_md, precision=4)

    stability_df = None
    if args.save_stability_metrics:
        stability_df = build_stability_metrics_table(
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            policy_score=policy_score,
            outcome_idx=utility_outcome_idx,
            constant_policy_quantile=float(args.constant_policy_quantile),
        )
        stability_csv = out_dir / "stability_metrics.csv"
        stability_md = out_dir / "stability_metrics.md"
        stability_df.to_csv(stability_csv, index=False)
        write_plain_markdown_table(stability_df, stability_md, precision=4)

    utility_df = None
    if args.save_utility_metrics:
        utility_df = build_utility_metrics_table(
            predictions_by_model=predictions_by_model,
            y_true=y_true,
            y_last_obs=test_raw.y_hist[:, -1, :],
            metadata_df=test_raw.metadata,
            outcome_idx=utility_outcome_idx,
            threshold_value=float(args.utility_threshold),
            risk_positive_quantile=float(args.risk_positive_quantile),
            actionable_rise_delta=float(args.actionable_rise_delta),
        )
        utility_csv = out_dir / "utility_metrics.csv"
        utility_md = out_dir / "utility_metrics.md"
        utility_df.to_csv(utility_csv, index=False)
        write_plain_markdown_table(utility_df, utility_md, precision=4)

    pd.DataFrame(
        {
            "window_index": np.arange(len(policy_score)),
            "policy_change_score": policy_score,
            "is_policy_change_subset": policy_mask.astype(int),
        }
    ).to_csv(out_dir / "policy_change_subset_windows.csv", index=False)
    if country_exclusion_df is not None:
        country_exclusion_df.to_csv(out_dir / "country_exclusion_summary.csv", index=False)
        (out_dir / "countries_dropped_negligible.txt").write_text(
            "\n".join(sorted(excluded_countries)) + ("\n" if excluded_countries else ""),
            encoding="utf-8",
        )

    write_metadata(
        out_path=out_dir / "evaluation_metadata.json",
        buckets=buckets,
        policy_change_method=args.policy_change_method,
        policy_change_quantile=args.policy_change_quantile,
        policy_change_threshold=args.policy_change_threshold,
        clip_nonnegative=args.clip_nonnegative,
        policy_subset_n=policy_subset_n,
        policy_subset_ratio=policy_subset_ratio,
        model_labels=[r["model"] for r in metrics_rows],
        outcome_cols=outcome_cols,
        policy_cols=policy_cols,
        country_filter=country_filter_meta,
    )

    print(f"Saved extended evaluation outputs to {out_dir}")
    print("Generated:")
    print(f"  - {out_dir / 'short_term_metrics.csv'}")
    print(f"  - {out_dir / 'short_term_metrics.md'}")
    print(f"  - {out_dir / 'long_term_metrics.csv'}")
    print(f"  - {out_dir / 'long_term_metrics.md'}")
    print(f"  - {out_dir / 'all_metrics_full.csv'}")
    if horizon_profile_ok:
        print(f"  - {horizon_profile_path}")
    print(f"  - {main_table_csv}")
    print(f"  - {main_table_md}")
    if persistence_cmp_df is not None:
        print(f"  - {out_dir / 'planning_vs_persistence.csv'}")
        print(f"  - {out_dir / 'planning_vs_persistence.md'}")
        print(f"  - {policy_subset_table_csv}")
        print(f"  - {policy_subset_table_md}")
    if regime_df is not None:
        print(f"  - {out_dir / 'regime_breakdown_metrics.csv'}")
        print(f"  - {out_dir / 'regime_breakdown_metrics.md'}")
        print(f"  - {out_dir / 'table_regime_breakdown.csv'}")
        print(f"  - {out_dir / 'table_regime_breakdown.md'}")
    if incidence_df is not None:
        print(f"  - {out_dir / 'incidence_regime_metrics.csv'}")
        print(f"  - {out_dir / 'incidence_regime_metrics.md'}")
    if shape_df is not None:
        print(f"  - {out_dir / 'shape_metrics.csv'}")
        print(f"  - {out_dir / 'shape_metrics.md'}")
    if stability_df is not None:
        print(f"  - {out_dir / 'stability_metrics.csv'}")
        print(f"  - {out_dir / 'stability_metrics.md'}")
    if utility_df is not None:
        print(f"  - {out_dir / 'utility_metrics.csv'}")
        print(f"  - {out_dir / 'utility_metrics.md'}")
    for p in trajectory_paths:
        print(f"  - {p}")
    if country_exclusion_df is not None:
        print(f"  - {out_dir / 'country_exclusion_summary.csv'}")
        print(f"  - {out_dir / 'countries_dropped_negligible.txt'}")
    print(f"  - {out_dir / 'policy_change_subset_windows.csv'}")
    print(f"  - {out_dir / 'evaluation_metadata.json'}")


if __name__ == "__main__":
    main()
