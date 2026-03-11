"""Train/test splits for panel windows."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def time_split_window_indices(
    metadata: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, pd.Timestamp]]:
    """Split windows by future start date: train = early dates, test = late dates."""
    if metadata.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}
    train_idx_list = []
    test_idx_list = []
    cutoff_dates = {}
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
        train_idx_list.extend(train_group.index.tolist())
        test_idx_list.extend(test_group.index.tolist())
    return (
        np.array(sorted(train_idx_list), dtype=np.int64),
        np.array(sorted(test_idx_list), dtype=np.int64),
        cutoff_dates,
    )
