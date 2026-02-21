from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd


_REQUIRED_PANEL_COLS = ("country", "date", "t_idx")


def _normalise_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_column_name(df: pd.DataFrame, requested: str) -> str:
    """Resolve a requested column name with light normalization fallback."""
    if requested in df.columns:
        return requested

    normalized_map = {
        _normalise_col_name(col): col
        for col in df.columns
    }
    normalized_requested = _normalise_col_name(requested)
    if normalized_requested in normalized_map:
        return normalized_map[normalized_requested]

    raise KeyError(f"Column '{requested}' was not found in dataframe")


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse mixed-format date columns and return timezone-naive timestamps."""
    if pd.api.types.is_datetime64_any_dtype(raw):
        parsed = pd.to_datetime(raw, errors="coerce")
        return parsed.dt.tz_localize(None)

    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        as_int = numeric.astype("Int64")
        as_str = as_int.astype(str)

        parsed = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
        if parsed.notna().any():
            return parsed.dt.tz_localize(None)

        fallback = pd.to_datetime(numeric, errors="coerce")
        return fallback.dt.tz_localize(None)

    parsed = pd.to_datetime(raw.astype(str), errors="coerce")
    return parsed.dt.tz_localize(None)


def load_oxford_csv(path: str | Path) -> pd.DataFrame:
    """Load an Oxford-style panel CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Oxford CSV not found: {csv_path}. "
            "Build it first with: python download_oxford_panel.py"
        )
    return pd.read_csv(csv_path)


def clean_oxford(
    df: pd.DataFrame,
    country_col: str = "CountryName",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Standard cleaning for Oxford panel data.

    - Parses date into pandas timestamp
    - Creates integer time index `t_idx`
    - Drops invalid (country, date) rows
    - Sorts by country/date and de-duplicates exact country-date duplicates
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    out = df.copy()
    resolved_country = _resolve_column_name(out, country_col)
    resolved_date = _resolve_column_name(out, date_col)

    out["country"] = out[resolved_country].astype(str).str.strip()
    out.loc[out["country"].isin(["", "nan", "None"]), "country"] = pd.NA

    out["date"] = _parse_dates(out[resolved_date])

    out = out.dropna(subset=["country", "date"])
    out = out.sort_values(["country", "date"])
    out = out.drop_duplicates(subset=["country", "date"], keep="last")

    unique_dates = pd.Index(sorted(out["date"].unique()))
    date_to_idx = {dt: idx for idx, dt in enumerate(unique_dates)}
    out["t_idx"] = out["date"].map(date_to_idx).astype("int64")

    return out


def select_features(
    df: pd.DataFrame,
    policy_cols: Iterable[str],
    outcome_cols: Iterable[str],
    state_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Select and validate policy/outcome/state columns.

    Returns dataframe with standardized panel columns plus requested features.
    Requested names are preserved in the output through column renaming.
    """
    for col in _REQUIRED_PANEL_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing required panel column '{col}'. Run clean_oxford() first.")

    policy_cols = list(policy_cols)
    outcome_cols = list(outcome_cols)
    state_cols = list(state_cols)

    requested_features = policy_cols + outcome_cols + state_cols
    resolved: List[str] = []
    rename_map: Dict[str, str] = {}

    for requested in requested_features:
        resolved_name = _resolve_column_name(df, requested)
        resolved.append(resolved_name)
        rename_map[resolved_name] = requested

    keep_cols = list(_REQUIRED_PANEL_COLS) + resolved
    selected = df[keep_cols].copy().rename(columns=rename_map)

    for feature in requested_features:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")

    return selected


def panel_stats(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[Dict[str, object], pd.Series]:
    """Return summary stats and per-feature missingness for quick diagnostics."""
    if df.empty:
        raise ValueError("Cannot compute stats on an empty panel")

    features = list(feature_cols)
    stats = {
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()),
        "date_min": pd.Timestamp(df["date"].min()).date().isoformat(),
        "date_max": pd.Timestamp(df["date"].max()).date().isoformat(),
    }

    missingness = df[features].isna().mean().sort_values(ascending=False) if features else pd.Series(dtype=float)
    return stats, missingness
