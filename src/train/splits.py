from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def time_split_panel(
    panel_df: pd.DataFrame,
    train_fraction: float = 0.8,
    country_col: str = "country",
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    """Split each country's timeline into early-train and late-test rows."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    cutoffs: Dict[str, pd.Timestamp] = {}

    for country, group in panel_df.groupby(country_col, sort=True):
        group = group.sort_values(date_col)
        n = len(group)
        if n < 2:
            continue

        cut_idx = max(1, int(np.floor(n * train_fraction)))
        cut_idx = min(cut_idx, n - 1)

        train = group.iloc[:cut_idx]
        test = group.iloc[cut_idx:]

        cutoffs[str(country)] = pd.Timestamp(train[date_col].max())
        train_parts.append(train)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else panel_df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else panel_df.iloc[0:0].copy()
    return train_df, test_df, cutoffs


def heldout_country_split_panel(
    panel_df: pd.DataFrame,
    holdout_fraction: float = 0.2,
    seed: int = 42,
    country_col: str = "country",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Randomly hold out a subset of countries for test evaluation."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")

    countries = np.array(sorted(panel_df[country_col].dropna().astype(str).unique()))
    if countries.size < 2:
        raise ValueError("Need at least two countries for held-out-country split")

    rng = np.random.default_rng(seed)
    n_holdout = int(np.ceil(countries.size * holdout_fraction))
    n_holdout = max(1, min(n_holdout, countries.size - 1))

    holdout = set(rng.choice(countries, size=n_holdout, replace=False).tolist())
    test_mask = panel_df[country_col].astype(str).isin(holdout)

    train_df = panel_df[~test_mask].reset_index(drop=True)
    test_df = panel_df[test_mask].reset_index(drop=True)
    return train_df, test_df, sorted(holdout)


def time_split_window_indices(
    metadata: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, pd.Timestamp]]:
    """
    Split pre-built windows by future start date within each country.

    Train windows use early future dates; test windows use later dates.
    """
    if metadata.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    train_indices: List[int] = []
    test_indices: List[int] = []
    cutoff_dates: Dict[str, pd.Timestamp] = {}

    for country, group in metadata.groupby("country", sort=True):
        group_sorted = group.sort_values("fut_start_date")
        unique_dates = np.array(sorted(group_sorted["fut_start_date"].unique()))

        if unique_dates.size < 2:
            continue

        cut_pos = int(np.floor(unique_dates.size * train_fraction)) - 1
        cut_pos = max(0, min(cut_pos, unique_dates.size - 2))
        cutoff = pd.Timestamp(unique_dates[cut_pos])
        cutoff_dates[str(country)] = cutoff

        train_group = group_sorted[group_sorted["fut_start_date"] <= cutoff]
        test_group = group_sorted[group_sorted["fut_start_date"] > cutoff]

        train_indices.extend(train_group.index.tolist())
        test_indices.extend(test_group.index.tolist())

    return (
        np.array(sorted(train_indices), dtype=np.int64),
        np.array(sorted(test_indices), dtype=np.int64),
        cutoff_dates,
    )


def heldout_country_window_indices(
    metadata: pd.DataFrame,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Split pre-built windows by held-out countries."""
    if metadata.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), []

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")

    countries = np.array(sorted(metadata["country"].astype(str).unique()))
    if countries.size < 2:
        raise ValueError("Need at least two countries for held-out-country split")

    rng = np.random.default_rng(seed)
    n_holdout = int(np.ceil(countries.size * holdout_fraction))
    n_holdout = max(1, min(n_holdout, countries.size - 1))

    holdout = set(rng.choice(countries, size=n_holdout, replace=False).tolist())

    train_idx = metadata[~metadata["country"].astype(str).isin(holdout)].index.to_numpy(dtype=np.int64)
    test_idx = metadata[metadata["country"].astype(str).isin(holdout)].index.to_numpy(dtype=np.int64)

    return np.sort(train_idx), np.sort(test_idx), sorted(holdout)
