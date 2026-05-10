"""Oxford panel loading and cleaning."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _normalise_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_column_name(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    normalized_map = {_normalise_col_name(col): col for col in df.columns}
    normalized_requested = _normalise_col_name(requested)
    if normalized_requested in normalized_map:
        return normalized_map[normalized_requested]
    raise KeyError(f"Column '{requested}' was not found in dataframe")


def _parse_dates(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(raw):
        parsed = pd.to_datetime(raw, errors="coerce")
        return parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        as_int = numeric.astype("Int64")
        as_str = as_int.astype(str)
        parsed = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
        if parsed.notna().any():
            return parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed
    parsed = pd.to_datetime(raw.astype(str), errors="coerce")
    return parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed


def load_oxford_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Oxford CSV not found: {path}")
    return pd.read_csv(path)


def clean_oxford(
    df: pd.DataFrame,
    country_col: str = "CountryName",
    date_col: str = "Date",
    country_code_col: str = "CountryCode",
) -> pd.DataFrame:
    out = df.copy()
    resolved_country = _resolve_column_name(out, country_col)
    resolved_date = _resolve_column_name(out, date_col)
    out["country"] = out[resolved_country].astype(str).str.strip()
    out.loc[out["country"].isin(["", "nan", "None"]), "country"] = pd.NA
    try:
        resolved_country_code = _resolve_column_name(out, country_code_col)
        out["country_code"] = out[resolved_country_code].astype(str).str.strip().str.upper()
        out.loc[out["country_code"].isin(["", "NAN", "NONE"]), "country_code"] = pd.NA
    except KeyError:
        out["country_code"] = pd.NA
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
    required = ["country", "date", "t_idx"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Run clean_oxford() first.")
    policy_cols = list(policy_cols)
    outcome_cols = list(outcome_cols)
    state_cols = list(state_cols)
    requested = policy_cols + outcome_cols + state_cols
    resolved = []
    rename_map = {}
    for req in requested:
        res = _resolve_column_name(df, req)
        resolved.append(res)
        rename_map[res] = req
    keep = required + resolved
    selected = df[keep].copy().rename(columns=rename_map)
    for feat in requested:
        selected[feat] = pd.to_numeric(selected[feat], errors="coerce")
    return selected


def panel_stats(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[Dict, pd.Series]:
    features = list(feature_cols)
    stats = {
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()),
        "date_min": pd.Timestamp(df["date"].min()).date().isoformat(),
        "date_max": pd.Timestamp(df["date"].max()).date().isoformat(),
    }
    missingness = df[features].isna().mean().sort_values(ascending=False) if features else pd.Series(dtype=float)
    return stats, missingness


def load_country_context_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Country context CSV not found: {path}")
    return pd.read_csv(path)


def merge_country_context(
    panel_df: pd.DataFrame,
    context_df: pd.DataFrame,
    context_cols: Iterable[str],
    panel_country_col: str = "country",
    panel_country_code_col: str = "country_code",
    context_country_col: str = "country",
    context_country_code_col: str = "CountryCode",
    zscore: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge static country-level context features into panel_df.

    Join strategy:
      1) By country code (preferred)
      2) Fallback by normalized country name for unmatched rows

    Context columns are median-imputed and optionally z-scored across countries.
    """
    ctx_cols_req = list(context_cols)
    if not ctx_cols_req:
        return panel_df.copy(), {"context_cols": 0, "rows_with_context": 0, "rows_total": int(len(panel_df))}

    ctx = context_df.copy()
    resolved_country = _resolve_column_name(ctx, context_country_col)
    resolved_code = _resolve_column_name(ctx, context_country_code_col)
    resolved_ctx = [_resolve_column_name(ctx, c) for c in ctx_cols_req]

    work = ctx[[resolved_country, resolved_code] + resolved_ctx].copy()
    rename = {resolved_country: "ctx_country", resolved_code: "ctx_country_code"}
    rename.update({res: req for res, req in zip(resolved_ctx, ctx_cols_req)})
    work = work.rename(columns=rename)

    work["ctx_country"] = work["ctx_country"].astype(str).str.strip()
    work["ctx_country_norm"] = work["ctx_country"].str.lower()
    work["ctx_country_code"] = work["ctx_country_code"].astype(str).str.strip().str.upper()
    work.loc[work["ctx_country_code"].isin(["", "NAN", "NONE"]), "ctx_country_code"] = pd.NA

    for c in ctx_cols_req:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        med = work[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        work[c] = work[c].fillna(float(med))
        if zscore:
            mu = float(work[c].mean())
            sd = float(work[c].std(ddof=0))
            sd = sd if sd > 1e-8 else 1.0
            work[c] = (work[c] - mu) / sd

    code_map = (
        work.dropna(subset=["ctx_country_code"])
        .drop_duplicates(subset=["ctx_country_code"], keep="last")
        .set_index("ctx_country_code")[ctx_cols_req]
    )
    name_map = (
        work.dropna(subset=["ctx_country_norm"])
        .drop_duplicates(subset=["ctx_country_norm"], keep="last")
        .set_index("ctx_country_norm")[ctx_cols_req]
    )

    out = panel_df.copy()
    if panel_country_code_col not in out.columns:
        out[panel_country_code_col] = pd.NA
    if panel_country_col not in out.columns:
        raise KeyError(f"Panel dataframe missing country column '{panel_country_col}'.")

    out["_join_code"] = out[panel_country_code_col].astype(str).str.strip().str.upper()
    out.loc[out["_join_code"].isin(["", "NAN", "NONE"]), "_join_code"] = pd.NA
    out["_join_name"] = out[panel_country_col].astype(str).str.strip().str.lower()

    merged = out.merge(
        code_map.reset_index().rename(columns={"ctx_country_code": "_join_code"}),
        on="_join_code",
        how="left",
    )

    missing_mask = merged[ctx_cols_req].isna().all(axis=1)
    if bool(missing_mask.any()):
        fallback = (
            name_map.reset_index()
            .rename(columns={"ctx_country_norm": "_join_name"})
        )
        miss = merged.loc[missing_mask, ["_join_name"]].merge(fallback, on="_join_name", how="left")
        for c in ctx_cols_req:
            merged.loc[missing_mask, c] = miss[c].to_numpy()

    for c in ctx_cols_req:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0.0)

    merged = merged.drop(columns=["_join_code", "_join_name"])
    rows_total = int(len(merged))
    rows_with_context = int((~merged[ctx_cols_req].isna().all(axis=1)).sum())
    stats = {
        "context_cols": int(len(ctx_cols_req)),
        "rows_with_context": rows_with_context,
        "rows_total": rows_total,
        "coverage": float(rows_with_context / rows_total) if rows_total > 0 else 0.0,
    }
    return merged, stats
