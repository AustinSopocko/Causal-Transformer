from __future__ import annotations

from dataclasses import dataclass

import torch

from .panel_windows import PanelWindows


@dataclass
class OutcomeScaler:
    mean: torch.Tensor
    std: torch.Tensor
    log1p: bool = False


def _safe_log1p(values: torch.Tensor) -> torch.Tensor:
    # Guard against tiny negatives that can appear after preprocessing.
    return torch.log1p(torch.clamp(values, min=-0.999999))


def fit_outcome_scaler(train_windows: PanelWindows, log1p: bool = False) -> OutcomeScaler:
    """Fit global outcome normalization stats on training windows only."""
    y_hist = train_windows.y_hist
    y_fut = train_windows.y_fut
    y_all = torch.cat([y_hist.reshape(-1, y_hist.shape[-1]), y_fut.reshape(-1, y_fut.shape[-1])], dim=0)

    if log1p:
        y_all = _safe_log1p(y_all)

    mean = torch.nanmean(y_all, dim=0)
    centered = y_all - mean
    centered = torch.where(torch.isnan(centered), torch.zeros_like(centered), centered)
    valid_counts = torch.sum(~torch.isnan(y_all), dim=0).clamp(min=1)
    var = torch.sum(centered ** 2, dim=0) / valid_counts
    std = torch.sqrt(var)
    std = torch.clamp(std, min=1e-6)

    return OutcomeScaler(mean=mean, std=std, log1p=log1p)


def transform_outcomes(values: torch.Tensor, scaler: OutcomeScaler) -> torch.Tensor:
    y = _safe_log1p(values) if scaler.log1p else values
    return (y - scaler.mean) / scaler.std


def inverse_transform_outcomes(values: torch.Tensor, scaler: OutcomeScaler) -> torch.Tensor:
    y = values * scaler.std + scaler.mean
    if scaler.log1p:
        y = torch.expm1(y)
    return y


def apply_outcome_scaler(windows: PanelWindows, scaler: OutcomeScaler) -> PanelWindows:
    """Return a new PanelWindows object with normalized outcome tensors."""
    return PanelWindows(
        x_hist=windows.x_hist,
        a_hist=windows.a_hist,
        y_hist=transform_outcomes(windows.y_hist, scaler),
        a_fut=windows.a_fut,
        y_fut=transform_outcomes(windows.y_fut, scaler),
        country_idx=windows.country_idx,
        metadata=windows.metadata,
        m_hist=windows.m_hist,
        dt_hist=windows.dt_hist,
    )
