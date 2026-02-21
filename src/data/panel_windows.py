from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PanelWindows:
    x_hist: torch.Tensor
    a_hist: torch.Tensor
    y_hist: torch.Tensor
    a_fut: torch.Tensor
    y_fut: torch.Tensor
    country_idx: torch.Tensor
    metadata: pd.DataFrame
    m_hist: Optional[torch.Tensor] = None
    dt_hist: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.x_hist.shape[0])

    def subset(self, indices: np.ndarray | List[int]) -> "PanelWindows":
        idx = torch.as_tensor(indices, dtype=torch.long)
        subset_meta = self.metadata.iloc[idx.tolist()].reset_index(drop=True)
        return PanelWindows(
            x_hist=self.x_hist[idx],
            a_hist=self.a_hist[idx],
            y_hist=self.y_hist[idx],
            a_fut=self.a_fut[idx],
            y_fut=self.y_fut[idx],
            country_idx=self.country_idx[idx],
            metadata=subset_meta,
            m_hist=self.m_hist[idx] if self.m_hist is not None else None,
            dt_hist=self.dt_hist[idx] if self.dt_hist is not None else None,
        )


class OxfordPanelDataset(Dataset):
    """Dataset wrapper around pre-built panel windows."""

    def __init__(self, windows: PanelWindows):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "x_hist": self.windows.x_hist[idx],
            "a_hist": self.windows.a_hist[idx],
            "y_hist": self.windows.y_hist[idx],
            "a_fut": self.windows.a_fut[idx],
            "y_fut": self.windows.y_fut[idx],
            "country_idx": self.windows.country_idx[idx],
        }
        if self.windows.m_hist is not None:
            sample["m_hist"] = self.windows.m_hist[idx]
        if self.windows.dt_hist is not None:
            sample["dt_hist"] = self.windows.dt_hist[idx]
        return sample


def make_windows(
    df: pd.DataFrame,
    history_len: int,
    forecast_horizon: int,
    stride: int,
    policy_cols: Iterable[str],
    outcome_cols: Iterable[str],
    state_cols: Iterable[str],
    country_to_idx: Optional[Dict[str, int]] = None,
    country_col: str = "country",
    date_col: str = "date",
    include_missing_mask: bool = False,
    include_time_delta: bool = False,
    drop_nan_windows: bool = False,
) -> PanelWindows:
    """Convert a country-date panel to rolling history/future windows."""
    if history_len <= 0 or forecast_horizon <= 0:
        raise ValueError("history_len and forecast_horizon must both be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    policy_cols = list(policy_cols)
    outcome_cols = list(outcome_cols)
    state_cols = list(state_cols)

    required_cols = [country_col, date_col] + policy_cols + outcome_cols + state_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for windowing: {missing}")

    if country_to_idx is None:
        countries = sorted(df[country_col].dropna().astype(str).unique())
        country_to_idx = {country: idx for idx, country in enumerate(countries)}

    panel = df.copy()
    panel[country_col] = panel[country_col].astype(str)
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
    panel = panel.dropna(subset=[country_col, date_col])
    panel = panel.sort_values([country_col, date_col])

    x_hist_rows: List[np.ndarray] = []
    a_hist_rows: List[np.ndarray] = []
    y_hist_rows: List[np.ndarray] = []
    a_fut_rows: List[np.ndarray] = []
    y_fut_rows: List[np.ndarray] = []
    country_idx_rows: List[int] = []

    mask_rows: List[np.ndarray] = []
    dt_rows: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []

    seq_len = history_len + forecast_horizon

    for country, group in panel.groupby(country_col, sort=True):
        if country not in country_to_idx:
            continue

        group = group.sort_values(date_col).reset_index(drop=True)
        n_steps = len(group)
        if n_steps < seq_len:
            continue

        state_values = (
            group[state_cols].to_numpy(dtype=np.float32)
            if state_cols
            else np.zeros((n_steps, 0), dtype=np.float32)
        )
        action_values = group[policy_cols].to_numpy(dtype=np.float32)
        outcome_values = group[outcome_cols].to_numpy(dtype=np.float32)
        date_values = group[date_col].to_numpy(dtype="datetime64[ns]")

        for start_idx in range(0, n_steps - seq_len + 1, stride):
            hist_slice = slice(start_idx, start_idx + history_len)
            fut_slice = slice(start_idx + history_len, start_idx + seq_len)

            x_hist = state_values[hist_slice]
            a_hist = action_values[hist_slice]
            y_hist = outcome_values[hist_slice]
            a_fut = action_values[fut_slice]
            y_fut = outcome_values[fut_slice]

            if drop_nan_windows:
                has_nan = (
                    np.isnan(x_hist).any()
                    or np.isnan(a_hist).any()
                    or np.isnan(y_hist).any()
                    or np.isnan(a_fut).any()
                    or np.isnan(y_fut).any()
                )
                if has_nan:
                    continue

            x_hist_rows.append(x_hist)
            a_hist_rows.append(a_hist)
            y_hist_rows.append(y_hist)
            a_fut_rows.append(a_fut)
            y_fut_rows.append(y_fut)
            country_idx_rows.append(country_to_idx[country])

            hist_dates = date_values[hist_slice]
            fut_dates = date_values[fut_slice]

            metadata_rows.append(
                {
                    "country": country,
                    "country_idx": country_to_idx[country],
                    "hist_start_date": pd.Timestamp(hist_dates[0]),
                    "hist_end_date": pd.Timestamp(hist_dates[-1]),
                    "fut_start_date": pd.Timestamp(fut_dates[0]),
                    "fut_end_date": pd.Timestamp(fut_dates[-1]),
                }
            )

            if include_missing_mask:
                mask_rows.append((~np.isnan(y_hist)).astype(np.float32))

            if include_time_delta:
                dt_hist = np.zeros(history_len, dtype=np.float32)
                if history_len > 1:
                    diffs = np.diff(hist_dates).astype("timedelta64[D]").astype(np.float32)
                    dt_hist[1:] = diffs
                dt_rows.append(dt_hist)

    d_x = len(state_cols)
    d_a = len(policy_cols)
    d_y = len(outcome_cols)

    if not x_hist_rows:
        empty_meta = pd.DataFrame(
            columns=[
                "country",
                "country_idx",
                "hist_start_date",
                "hist_end_date",
                "fut_start_date",
                "fut_end_date",
            ]
        )
        return PanelWindows(
            x_hist=torch.empty((0, history_len, d_x), dtype=torch.float32),
            a_hist=torch.empty((0, history_len, d_a), dtype=torch.float32),
            y_hist=torch.empty((0, history_len, d_y), dtype=torch.float32),
            a_fut=torch.empty((0, forecast_horizon, d_a), dtype=torch.float32),
            y_fut=torch.empty((0, forecast_horizon, d_y), dtype=torch.float32),
            country_idx=torch.empty((0,), dtype=torch.long),
            metadata=empty_meta,
            m_hist=torch.empty((0, history_len, d_y), dtype=torch.float32) if include_missing_mask else None,
            dt_hist=torch.empty((0, history_len), dtype=torch.float32) if include_time_delta else None,
        )

    metadata = pd.DataFrame(metadata_rows)

    windows = PanelWindows(
        x_hist=torch.from_numpy(np.stack(x_hist_rows).astype(np.float32)),
        a_hist=torch.from_numpy(np.stack(a_hist_rows).astype(np.float32)),
        y_hist=torch.from_numpy(np.stack(y_hist_rows).astype(np.float32)),
        a_fut=torch.from_numpy(np.stack(a_fut_rows).astype(np.float32)),
        y_fut=torch.from_numpy(np.stack(y_fut_rows).astype(np.float32)),
        country_idx=torch.as_tensor(country_idx_rows, dtype=torch.long),
        metadata=metadata,
        m_hist=torch.from_numpy(np.stack(mask_rows).astype(np.float32)) if include_missing_mask else None,
        dt_hist=torch.from_numpy(np.stack(dt_rows).astype(np.float32)) if include_time_delta else None,
    )
    return windows


def describe_window_shapes(windows: PanelWindows) -> Dict[str, tuple]:
    summary = {
        "x_hist": tuple(windows.x_hist.shape),
        "a_hist": tuple(windows.a_hist.shape),
        "y_hist": tuple(windows.y_hist.shape),
        "a_fut": tuple(windows.a_fut.shape),
        "y_fut": tuple(windows.y_fut.shape),
        "country_idx": tuple(windows.country_idx.shape),
    }
    if windows.m_hist is not None:
        summary["m_hist"] = tuple(windows.m_hist.shape)
    if windows.dt_hist is not None:
        summary["dt_hist"] = tuple(windows.dt_hist.shape)
    return summary
