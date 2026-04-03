#!/usr/bin/env python
"""
Train CRT on Oxford panel dataset.

Usage:
    python train_oxford.py --oxford_csv data/oxford/oxford_panel.csv --config src/configs/oxford_config.yaml --model_type crt
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

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
from src.data.panel_windows import OxfordPanelDataset, build_country_index, make_windows
from src.train.splits import time_split_window_indices


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_config(
    cfg: dict,
    d_x: int,
    d_a: int,
    d_y: int,
    num_countries: int,
) -> CRTConfig:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    window_cfg = cfg["window"]
    return CRTConfig(
        d_x=d_x,
        d_a=d_a,
        d_y=d_y,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers_enc=int(model_cfg["n_layers_enc"]),
        n_layers_dec=int(model_cfg["n_layers_dec"]),
        history_len=int(window_cfg["history_len"]),
        forecast_horizon=int(window_cfg["forecast_horizon"]),
        dropout=float(model_cfg["dropout"]),
        lr=float(train_cfg["lr"]),
        teacher_forcing_start=float(train_cfg["teacher_forcing_start"]),
        teacher_forcing_end=float(train_cfg["teacher_forcing_end"]),
        num_countries=max(1, num_countries),
        use_country_context=bool(model_cfg.get("use_country_context", True)),
        use_future_policy=bool(model_cfg.get("use_future_policy", True)),
        target_mode=str(model_cfg.get("target_mode", "absolute")),
    )


def evaluate_rmse(
    model: nn.Module,
    loader: DataLoader,
    scaler: OutcomeScaler,
    device: str,
    long_start: Optional[int] = None,
    late_start: Optional[int] = None,
) -> Dict[str, float | List[float]]:
    model.eval()
    y_pred_all = []
    y_true_all = []
    with torch.no_grad():
        for batch in loader:
            x_hist = batch["x_hist"].to(device)
            a_hist = batch["a_hist"].to(device)
            y_hist = batch["y_hist"].to(device)
            a_fut = batch["a_fut"].to(device)
            y_fut = batch["y_fut"].to(device)
            country_idx = batch["country_idx"].to(device)

            y_pred = rollout(
                model=model,
                x_hist=x_hist,
                a_hist=a_hist,
                y_hist=y_hist,
                a_fut=a_fut,
                country_idx=country_idx,
            )
            y_pred_all.append(inverse_transform_outcomes(y_pred.cpu(), scaler))
            y_true_all.append(inverse_transform_outcomes(y_fut.cpu(), scaler))

    y_pred_full = torch.cat(y_pred_all, dim=0)
    y_true_full = torch.cat(y_true_all, dim=0)
    sq_err = (y_pred_full - y_true_full) ** 2
    overall = torch.sqrt(torch.mean(sq_err)).item()
    per_horizon = [torch.sqrt(torch.mean(sq_err[:, h, :])).item() for h in range(sq_err.shape[1])]
    out: Dict[str, float | List[float]] = {"overall": overall, "per_horizon": per_horizon}

    h = len(per_horizon)
    if long_start is not None:
        s = max(1, min(int(long_start), h))
        out["long_rmse"] = float(sum(per_horizon[s - 1 :]) / max(h - s + 1, 1))
    if late_start is not None:
        s = max(1, min(int(late_start), h))
        out["late_rmse"] = float(sum(per_horizon[s - 1 :]) / max(h - s + 1, 1))
    return out


def build_horizon_weights(
    horizon: int,
    weighting: str = "none",
    start: float = 1.0,
    end: float = 1.0,
    power: float = 2.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Build non-negative per-horizon weights for sequence loss."""
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if start <= 0 or end <= 0:
        raise ValueError(f"loss horizon weights must be > 0, got start={start}, end={end}")

    weighting = str(weighting).lower()
    if weighting in {"none", "uniform"}:
        weights = torch.ones(horizon, dtype=torch.float32, device=device)
    elif weighting == "linear":
        weights = torch.linspace(start, end, steps=horizon, dtype=torch.float32, device=device)
    elif weighting == "power":
        if horizon == 1:
            weights = torch.tensor([end], dtype=torch.float32, device=device)
        else:
            t = torch.linspace(0.0, 1.0, steps=horizon, dtype=torch.float32, device=device)
            weights = start + (end - start) * torch.pow(t, power)
    else:
        raise ValueError(f"Unknown loss.horizon_weighting='{weighting}'. Use one of: none, linear, power")

    return weights / torch.clamp(weights.mean(), min=1e-8)


def compute_sequence_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    loss_type: str,
    horizon_weights: torch.Tensor,
    huber_delta: float = 1.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute sequence loss with optional horizon weighting.
    y_pred/y_true: (B, H, d_y), horizon_weights: (H,)
    """
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        per_elem = (y_pred - y_true) ** 2
    elif loss_type == "mae":
        per_elem = torch.abs(y_pred - y_true)
    elif loss_type == "huber":
        per_elem = F.huber_loss(y_pred, y_true, delta=huber_delta, reduction="none")
    else:
        raise ValueError(f"Unknown loss.type='{loss_type}'. Use one of: mse, mae, huber")

    weighted = per_elem * horizon_weights.view(1, -1, 1)
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != weighted.shape[0]:
            raise ValueError(
                f"sample_weights must be shape (B,), got {tuple(sample_weights.shape)} for batch {weighted.shape[0]}"
            )
        weighted = weighted * sample_weights.view(-1, 1, 1).to(weighted.dtype)
    return torch.mean(weighted)


def build_policy_change_weights(
    a_fut: torch.Tensor,
    power: float = 0.0,
    max_weight: float = 0.0,
) -> torch.Tensor:
    """
    Compute per-sample weights from future policy volatility.
    Weights are normalized to mean=1 to preserve effective learning rate.
    """
    bsz = a_fut.shape[0]
    if power <= 0.0 or a_fut.shape[1] < 2:
        return torch.ones(bsz, dtype=torch.float32, device=a_fut.device)

    delta = torch.abs(a_fut[:, 1:, :] - a_fut[:, :-1, :])
    score = torch.mean(delta, dim=(1, 2))
    weights = torch.pow(1.0 + score, power)
    if max_weight > 0.0:
        weights = torch.clamp(weights, max=max_weight)
    return weights / torch.clamp(weights.mean(), min=1e-8)


def compute_step_deltas(y_seq: torch.Tensor, y_last_hist: torch.Tensor) -> torch.Tensor:
    """
    Convert absolute trajectories to step deltas.
    First delta is relative to history endpoint; subsequent deltas are step-to-step.
    """
    first = y_seq[:, :1, :] - y_last_hist
    if y_seq.shape[1] == 1:
        return first
    rest = y_seq[:, 1:, :] - y_seq[:, :-1, :]
    return torch.cat([first, rest], dim=1)


def compute_anchor_loss(y_pred: torch.Tensor, y_hist: torch.Tensor) -> torch.Tensor:
    """
    Soft continuity term at forecast start:
    penalize mismatch between first predicted step and history endpoint.
    """
    return F.mse_loss(y_pred[:, 0, :], y_hist[:, -1, :], reduction="mean")


def compute_nonnegative_penalty(
    y_pred_scaled: torch.Tensor,
    scaler: OutcomeScaler,
    log1p_outcomes: bool,
) -> torch.Tensor:
    """
    Penalize negative forecasts in original outcome space.
    """
    y_pred_orig = inverse_transform_outcomes(y_pred_scaled, scaler)
    if log1p_outcomes:
        y_pred_orig = torch.expm1(y_pred_orig)
    return torch.mean(torch.relu(-y_pred_orig) ** 2)


def teacher_forcing_ratio_for_epoch(
    epoch: int,
    epochs: int,
    start: float,
    end: float,
    schedule: str = "linear",
) -> float:
    """Compute teacher forcing ratio for an epoch index [0, epochs-1]."""
    if epochs <= 1:
        return float(end)
    p = float(epoch) / float(epochs - 1)
    schedule = str(schedule).lower()

    if schedule == "linear":
        alpha = p
    elif schedule == "cosine":
        alpha = 0.5 * (1.0 - math.cos(math.pi * p))
    else:
        raise ValueError(f"Unknown training.teacher_forcing_schedule='{schedule}'. Use: linear, cosine")
    return float(start + (end - start) * alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRT on Oxford panel")
    parser.add_argument("--oxford_csv", type=str, required=True, help="Path to Oxford panel CSV")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/oxford")
    parser.add_argument("--model_type", type=str, default="crt", choices=["crt"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    dataset_cfg = cfg["dataset"]
    window_cfg = cfg["window"]
    split_cfg = cfg.get("split", {})
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})
    norm_cfg = cfg.get("normalization", {})

    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 40))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 64))
    train_fraction = float(split_cfg.get("train_fraction", 0.8))
    tf_schedule = str(train_cfg.get("teacher_forcing_schedule", "linear"))
    rollout_loss_weight = float(train_cfg.get("rollout_loss_weight", 0.0))
    rollout_loss_start_epoch = int(train_cfg.get("rollout_loss_start_epoch", 0))
    policy_change_weight_power = float(train_cfg.get("policy_change_weight_power", 0.0))
    policy_change_weight_max = float(train_cfg.get("policy_change_weight_max", 0.0))
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "overall_rmse")).lower()
    checkpoint_long_start = int(train_cfg.get("checkpoint_long_start", 8))
    checkpoint_late_start = int(train_cfg.get("checkpoint_late_start", 12))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))

    loss_type = str(loss_cfg.get("type", "mse"))
    increment_loss_weight = float(loss_cfg.get("increment_loss_weight", 0.0))
    horizon_weighting = str(loss_cfg.get("horizon_weighting", "none"))
    horizon_weight_start = float(loss_cfg.get("horizon_weight_start", 1.0))
    horizon_weight_end = float(loss_cfg.get("horizon_weight_end", 1.0))
    horizon_weight_power = float(loss_cfg.get("horizon_weight_power", 2.0))
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))
    anchor_weight = float(loss_cfg.get("anchor_weight", 0.0))
    nonneg_weight = float(loss_cfg.get("nonneg_weight", 0.0))
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", split_cfg.get("seed", 42)))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    policy_cols = list(dataset_cfg["policy_cols"])
    outcome_cols = list(dataset_cfg["outcome_cols"])
    state_cols = list(dataset_cfg.get("state_cols", []))
    context_cols = list(dataset_cfg.get("context_cols", []))
    context_csv = dataset_cfg.get("country_context_csv", None)
    split_no_future_overlap = bool(split_cfg.get("no_future_overlap", False))

    raw = load_oxford_csv(args.oxford_csv)
    cleaned = clean_oxford(
        raw,
        country_col=dataset_cfg.get("country_col", "CountryName"),
        date_col=dataset_cfg.get("date_col", "Date"),
        country_code_col=dataset_cfg.get("country_code_col", "CountryCode"),
    )
    context_stats = None
    if context_cols:
        if not context_csv:
            raise ValueError(
                "dataset.context_cols is non-empty but dataset.country_context_csv is not set in config."
            )
        context_df = load_country_context_csv(context_csv)
        cleaned, context_stats = merge_country_context(
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
    # preserve order and avoid duplicate feature names
    state_cols = list(dict.fromkeys(state_cols))
    panel_df = select_features(cleaned, policy_cols=policy_cols, outcome_cols=outcome_cols, state_cols=state_cols)

    if not state_cols:
        panel_df["__dummy_state__"] = 0.0
        state_cols = ["__dummy_state__"]

    country_to_idx = build_country_index(panel_df)
    history_len = int(window_cfg["history_len"])
    forecast_horizon = int(window_cfg["forecast_horizon"])
    stride = int(window_cfg["stride"])
    checkpoint_long_start = int(max(1, min(checkpoint_long_start, forecast_horizon)))
    checkpoint_late_start = int(max(1, min(checkpoint_late_start, forecast_horizon)))
    valid_checkpoint_metrics = {"overall_rmse", "long_rmse", "late_rmse"}
    if checkpoint_metric not in valid_checkpoint_metrics:
        raise ValueError(
            f"Unknown training.checkpoint_metric='{checkpoint_metric}'. "
            f"Use one of: {sorted(valid_checkpoint_metrics)}"
        )

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

    if len(windows) == 0:
        raise RuntimeError("No windows produced. Check data and config.")

    train_idx, test_idx, _ = time_split_window_indices(
        windows.metadata,
        train_fraction=train_fraction,
        no_future_overlap=split_no_future_overlap,
    )
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError("Empty train or test split.")

    train_windows = windows.subset(train_idx)
    test_windows = windows.subset(test_idx)

    log1p = bool(norm_cfg.get("log1p_outcomes", False))
    scaler = fit_outcome_scaler(train_windows, log1p=log1p)
    train_windows = apply_outcome_scaler(train_windows, scaler, log1p=log1p)
    test_windows = apply_outcome_scaler(test_windows, scaler, log1p=log1p)

    model_cfg = build_model_config(
        cfg=cfg,
        d_x=len(state_cols),
        d_a=len(policy_cols),
        d_y=len(outcome_cols),
        num_countries=len(country_to_idx),
    )

    print("=" * 60)
    print("MODEL CONFIG")
    print("=" * 60)
    print(f"d_x={model_cfg.d_x}, d_a={model_cfg.d_a}, d_y={model_cfg.d_y}, num_countries={model_cfg.num_countries}")
    print(f"target_mode={getattr(model_cfg, 'target_mode', 'absolute')}")
    print(
        f"use_country_context={getattr(model_cfg, 'use_country_context', False)}, "
        f"use_future_policy={getattr(model_cfg, 'use_future_policy', True)}"
    )
    print(f"split.no_future_overlap={split_no_future_overlap}")
    if context_stats is not None:
        print(
            f"context_cols={context_cols} "
            f"(coverage={context_stats.get('coverage', 0.0):.3f}, "
            f"rows={context_stats.get('rows_with_context', 0)}/{context_stats.get('rows_total', 0)})"
        )
    print("=" * 60)

    model = CRTModel(model_cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr)
    horizon_weights = build_horizon_weights(
        horizon=model_cfg.forecast_horizon,
        weighting=horizon_weighting,
        start=horizon_weight_start,
        end=horizon_weight_end,
        power=horizon_weight_power,
        device=args.device,
    )

    print("TRAINING CONFIG")
    print(
        f"loss.type={loss_type}, loss.horizon_weighting={horizon_weighting}, "
        f"loss.increment_loss_weight={increment_loss_weight}, loss.anchor_weight={anchor_weight}, "
        f"loss.nonneg_weight={nonneg_weight}, "
        f"rollout_loss_weight={rollout_loss_weight}, rollout_start_epoch={rollout_loss_start_epoch}"
    )
    print(
        f"tf_schedule={tf_schedule}, tf_start={model_cfg.teacher_forcing_start}, "
        f"tf_end={model_cfg.teacher_forcing_end}, grad_clip_norm={grad_clip_norm}, "
        f"policy_change_weight_power={policy_change_weight_power}, policy_change_weight_max={policy_change_weight_max}"
    )
    print(
        f"checkpoint_metric={checkpoint_metric} "
        f"(long_start={checkpoint_long_start}, late_start={checkpoint_late_start})"
    )
    print(f"seed={seed}")
    print("=" * 60)

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(
        OxfordPanelDataset(train_windows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_gen,
    )
    test_loader = DataLoader(OxfordPanelDataset(test_windows), batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metric_value = float("inf")
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tf_loss = 0.0
        total_tf_level_loss = 0.0
        total_tf_increment_loss = 0.0
        total_tf_anchor_loss = 0.0
        total_tf_nonneg_loss = 0.0
        total_rollout_loss = 0.0
        total_rollout_level_loss = 0.0
        total_rollout_increment_loss = 0.0
        total_rollout_anchor_loss = 0.0
        total_rollout_nonneg_loss = 0.0
        tf_ratio = teacher_forcing_ratio_for_epoch(
            epoch=epoch,
            epochs=epochs,
            start=model_cfg.teacher_forcing_start,
            end=model_cfg.teacher_forcing_end,
            schedule=tf_schedule,
        )
        use_rollout_loss = rollout_loss_weight > 0.0 and (epoch + 1) >= rollout_loss_start_epoch

        for batch in train_loader:
            x_hist = batch["x_hist"].to(args.device)
            a_hist = batch["a_hist"].to(args.device)
            y_hist = batch["y_hist"].to(args.device)
            a_fut = batch["a_fut"].to(args.device)
            y_fut = batch["y_fut"].to(args.device)
            country_idx = batch["country_idx"].to(args.device)
            sample_weights = build_policy_change_weights(
                a_fut=a_fut,
                power=policy_change_weight_power,
                max_weight=policy_change_weight_max,
            )
            y_last_hist = y_hist[:, -1:, :]

            y_pred = model(
                x_hist=x_hist,
                a_hist=a_hist,
                y_hist=y_hist,
                a_fut=a_fut,
                y_fut=y_fut,
                teacher_forcing_prob=tf_ratio,
                country_idx=country_idx,
            )
            tf_level_loss = compute_sequence_loss(
                y_pred=y_pred,
                y_true=y_fut,
                loss_type=loss_type,
                horizon_weights=horizon_weights,
                huber_delta=huber_delta,
                sample_weights=sample_weights,
            )
            if increment_loss_weight > 0.0:
                tf_increment_loss = compute_sequence_loss(
                    y_pred=compute_step_deltas(y_pred, y_last_hist),
                    y_true=compute_step_deltas(y_fut, y_last_hist),
                    loss_type=loss_type,
                    horizon_weights=horizon_weights,
                    huber_delta=huber_delta,
                    sample_weights=sample_weights,
                )
            else:
                tf_increment_loss = torch.zeros((), device=args.device, dtype=tf_level_loss.dtype)
            if anchor_weight > 0.0:
                tf_anchor_loss = compute_anchor_loss(y_pred=y_pred, y_hist=y_hist)
            else:
                tf_anchor_loss = torch.zeros((), device=args.device, dtype=tf_level_loss.dtype)
            if nonneg_weight > 0.0:
                tf_nonneg_loss = compute_nonnegative_penalty(
                    y_pred_scaled=y_pred,
                    scaler=scaler,
                    log1p_outcomes=log1p,
                )
            else:
                tf_nonneg_loss = torch.zeros((), device=args.device, dtype=tf_level_loss.dtype)
            tf_loss = (
                tf_level_loss
                + increment_loss_weight * tf_increment_loss
                + anchor_weight * tf_anchor_loss
                + nonneg_weight * tf_nonneg_loss
            )

            if use_rollout_loss:
                y_rollout = model(
                    x_hist=x_hist,
                    a_hist=a_hist,
                    y_hist=y_hist,
                    a_fut=a_fut,
                    y_fut=None,
                    country_idx=country_idx,
                )
                rollout_level_loss = compute_sequence_loss(
                    y_pred=y_rollout,
                    y_true=y_fut,
                    loss_type=loss_type,
                    horizon_weights=horizon_weights,
                    huber_delta=huber_delta,
                    sample_weights=sample_weights,
                )
                if increment_loss_weight > 0.0:
                    rollout_increment_loss = compute_sequence_loss(
                        y_pred=compute_step_deltas(y_rollout, y_last_hist),
                        y_true=compute_step_deltas(y_fut, y_last_hist),
                        loss_type=loss_type,
                        horizon_weights=horizon_weights,
                        huber_delta=huber_delta,
                        sample_weights=sample_weights,
                    )
                else:
                    rollout_increment_loss = torch.zeros((), device=args.device, dtype=rollout_level_loss.dtype)
                if anchor_weight > 0.0:
                    rollout_anchor_loss = compute_anchor_loss(y_pred=y_rollout, y_hist=y_hist)
                else:
                    rollout_anchor_loss = torch.zeros((), device=args.device, dtype=rollout_level_loss.dtype)
                if nonneg_weight > 0.0:
                    rollout_nonneg_loss = compute_nonnegative_penalty(
                        y_pred_scaled=y_rollout,
                        scaler=scaler,
                        log1p_outcomes=log1p,
                    )
                else:
                    rollout_nonneg_loss = torch.zeros((), device=args.device, dtype=rollout_level_loss.dtype)
                rollout_loss = (
                    rollout_level_loss
                    + increment_loss_weight * rollout_increment_loss
                    + anchor_weight * rollout_anchor_loss
                    + nonneg_weight * rollout_nonneg_loss
                )
                loss = (1.0 - rollout_loss_weight) * tf_loss + rollout_loss_weight * rollout_loss
            else:
                rollout_level_loss = torch.zeros((), device=args.device, dtype=tf_loss.dtype)
                rollout_increment_loss = torch.zeros((), device=args.device, dtype=tf_loss.dtype)
                rollout_anchor_loss = torch.zeros((), device=args.device, dtype=tf_loss.dtype)
                rollout_nonneg_loss = torch.zeros((), device=args.device, dtype=tf_loss.dtype)
                rollout_loss = torch.zeros((), device=args.device, dtype=tf_loss.dtype)
                loss = tf_loss

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            total_tf_loss += float(tf_loss.item())
            total_tf_level_loss += float(tf_level_loss.item())
            total_tf_increment_loss += float(tf_increment_loss.item())
            total_tf_anchor_loss += float(tf_anchor_loss.item())
            total_tf_nonneg_loss += float(tf_nonneg_loss.item())
            total_rollout_loss += float(rollout_loss.item())
            total_rollout_level_loss += float(rollout_level_loss.item())
            total_rollout_increment_loss += float(rollout_increment_loss.item())
            total_rollout_anchor_loss += float(rollout_anchor_loss.item())
            total_rollout_nonneg_loss += float(rollout_nonneg_loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        train_tf_loss = total_tf_loss / max(len(train_loader), 1)
        train_tf_level_loss = total_tf_level_loss / max(len(train_loader), 1)
        train_tf_increment_loss = total_tf_increment_loss / max(len(train_loader), 1)
        train_tf_anchor_loss = total_tf_anchor_loss / max(len(train_loader), 1)
        train_tf_nonneg_loss = total_tf_nonneg_loss / max(len(train_loader), 1)
        train_rollout_loss = total_rollout_loss / max(len(train_loader), 1)
        train_rollout_level_loss = total_rollout_level_loss / max(len(train_loader), 1)
        train_rollout_increment_loss = total_rollout_increment_loss / max(len(train_loader), 1)
        train_rollout_anchor_loss = total_rollout_anchor_loss / max(len(train_loader), 1)
        train_rollout_nonneg_loss = total_rollout_nonneg_loss / max(len(train_loader), 1)
        rmse = evaluate_rmse(
            model,
            test_loader,
            scaler,
            args.device,
            long_start=checkpoint_long_start,
            late_start=checkpoint_late_start,
        )
        long_rmse = float(rmse.get("long_rmse", rmse["overall"]))
        late_rmse = float(rmse.get("late_rmse", rmse["overall"]))
        metric_value = {
            "overall_rmse": float(rmse["overall"]),
            "long_rmse": long_rmse,
            "late_rmse": late_rmse,
        }[checkpoint_metric]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"tf_loss={train_tf_loss:.6f} "
            f"(level={train_tf_level_loss:.6f}, incr={train_tf_increment_loss:.6f}, "
            f"anchor={train_tf_anchor_loss:.6f}, nonneg={train_tf_nonneg_loss:.6f}) | "
            f"rollout_loss={train_rollout_loss:.6f} "
            f"(level={train_rollout_level_loss:.6f}, incr={train_rollout_increment_loss:.6f}, "
            f"anchor={train_rollout_anchor_loss:.6f}, nonneg={train_rollout_nonneg_loss:.6f}) | "
            f"test_rmse={rmse['overall']:.6f} | test_long_rmse={long_rmse:.6f} | test_late_rmse={late_rmse:.6f} | "
            f"checkpoint_metric={checkpoint_metric}:{metric_value:.6f} | "
            f"tf={tf_ratio:.3f}"
        )

        if metric_value < best_metric_value:
            best_metric_value = metric_value
            epochs_since_improvement = 0
            best_path = checkpoint_dir / f"best_{args.model_type}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                    "scaler": scaler,
                    "country_to_idx": country_to_idx,
                    "policy_cols": policy_cols,
                    "outcome_cols": outcome_cols,
                    "state_cols": state_cols,
                    "context_cols": context_cols,
                    "country_context_csv": str(context_csv) if context_csv else None,
                    "split_no_future_overlap": split_no_future_overlap,
                    "rmse": rmse,
                    "model_type": args.model_type,
                    "training": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "teacher_forcing_schedule": tf_schedule,
                        "seed": seed,
                        "teacher_forcing_start": model_cfg.teacher_forcing_start,
                        "teacher_forcing_end": model_cfg.teacher_forcing_end,
                        "rollout_loss_weight": rollout_loss_weight,
                        "rollout_loss_start_epoch": rollout_loss_start_epoch,
                        "policy_change_weight_power": policy_change_weight_power,
                        "policy_change_weight_max": policy_change_weight_max,
                        "checkpoint_metric": checkpoint_metric,
                        "checkpoint_long_start": checkpoint_long_start,
                        "checkpoint_late_start": checkpoint_late_start,
                        "loss_type": loss_type,
                        "increment_loss_weight": increment_loss_weight,
                        "horizon_weighting": horizon_weighting,
                        "horizon_weight_start": horizon_weight_start,
                        "horizon_weight_end": horizon_weight_end,
                        "horizon_weight_power": horizon_weight_power,
                        "huber_delta": huber_delta,
                        "anchor_weight": anchor_weight,
                        "nonneg_weight": nonneg_weight,
                        "target_mode": getattr(model_cfg, "target_mode", "absolute"),
                        "grad_clip_norm": grad_clip_norm,
                        "early_stopping_patience": early_stopping_patience,
                    },
                },
                best_path,
            )
            print(f"  Saved best to {best_path}")
        else:
            epochs_since_improvement += 1

        if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no {checkpoint_metric} improvement for {epochs_since_improvement} epochs)."
            )
            break

    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_rmse = evaluate_rmse(
        model,
        test_loader,
        scaler,
        args.device,
        long_start=checkpoint_long_start,
        late_start=checkpoint_late_start,
    )
    print(f"Overall RMSE: {final_rmse['overall']:.6f}")
    print(f"Long RMSE ({checkpoint_long_start}..H): {float(final_rmse.get('long_rmse', final_rmse['overall'])):.6f}")
    print(f"Late RMSE ({checkpoint_late_start}..H): {float(final_rmse.get('late_rmse', final_rmse['overall'])):.6f}")
    for idx, v in enumerate(final_rmse["per_horizon"], start=1):
        print(f"  Step {idx}: {v:.6f}")


if __name__ == "__main__":
    main()
