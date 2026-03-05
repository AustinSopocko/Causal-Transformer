from __future__ import annotations

import pandas as pd

from src.data.panel_windows import make_windows


def _build_panel() -> pd.DataFrame:
    rows = []
    for country_idx, country in enumerate(["A", "B"]):
        for day in range(10):
            rows.append(
                {
                    "country": country,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                    "policy": float(day),
                    "outcome": float(country_idx * 100 + day),
                    "state_1": float(country_idx),
                    "state_2": float(day * 2),
                }
            )
    return pd.DataFrame(rows)


def test_make_windows_shapes_and_alignment() -> None:
    df = _build_panel()
    windows = make_windows(
        df=df,
        history_len=3,
        forecast_horizon=2,
        stride=1,
        policy_cols=["policy"],
        outcome_cols=["outcome"],
        state_cols=["state_1", "state_2"],
        country_to_idx={"A": 0, "B": 1},
    )

    # Per-country windows: 10 - (3 + 2) + 1 = 6; total = 12
    assert len(windows) == 12
    assert tuple(windows.x_hist.shape) == (12, 3, 2)
    assert tuple(windows.a_hist.shape) == (12, 3, 1)
    assert tuple(windows.y_hist.shape) == (12, 3, 1)
    assert tuple(windows.a_fut.shape) == (12, 2, 1)
    assert tuple(windows.y_fut.shape) == (12, 2, 1)
    assert tuple(windows.country_idx.shape) == (12,)

    for row in windows.metadata.itertuples(index=False):
        expected_country_idx = 0 if row.country == "A" else 1
        assert row.country_idx == expected_country_idx
        assert row.fut_start_date == row.hist_end_date + pd.Timedelta(days=1)


def test_make_windows_respects_stride() -> None:
    df = _build_panel()
    windows = make_windows(
        df=df,
        history_len=3,
        forecast_horizon=2,
        stride=2,
        policy_cols=["policy"],
        outcome_cols=["outcome"],
        state_cols=["state_1", "state_2"],
        country_to_idx={"A": 0, "B": 1},
    )

    # Per-country start positions: 0,2,4 -> 3 windows; total = 6
    assert len(windows) == 6
