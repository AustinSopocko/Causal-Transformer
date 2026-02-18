#!/usr/bin/env python
"""
Download and build Oxford policy + OWID outcome panel dataset.

Usage:
    python download_oxford_panel.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OXFORD_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv"
OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"


OXFORD_COLUMNS = [
    "CountryName",
    "CountryCode",
    "Date",
    "StringencyIndex_Average",
    "C1M_School closing",
    "C2M_Workplace closing",
    "C6M_Stay at home requirements",
]

OWID_COLUMNS = [
    "iso_code",
    "date",
    "new_cases_smoothed_per_million",
    "new_deaths_smoothed_per_million",
]


def build_panel(output_path: Path) -> pd.DataFrame:
    print("Downloading Oxford policy data...")
    oxford = pd.read_csv(OXFORD_URL, usecols=OXFORD_COLUMNS)

    print("Downloading OWID outcome data...")
    owid = pd.read_csv(OWID_URL, usecols=OWID_COLUMNS)

    # Parse dates for merge key.
    oxford["date_dt"] = pd.to_datetime(oxford["Date"].astype(str), format="%Y%m%d", errors="coerce")
    owid["date_dt"] = pd.to_datetime(owid["date"], errors="coerce")

    # Keep only country ISO-3 rows from OWID (drop aggregates like OWID_WRL).
    owid = owid[owid["iso_code"].str.len() == 3].copy()
    owid = owid.dropna(subset=["iso_code", "date_dt"])
    owid = owid.sort_values(["iso_code", "date_dt"]).drop_duplicates(
        subset=["iso_code", "date_dt"],
        keep="last",
    )

    panel = oxford.merge(
        owid[[
            "iso_code",
            "date_dt",
            "new_cases_smoothed_per_million",
            "new_deaths_smoothed_per_million",
        ]],
        left_on=["CountryCode", "date_dt"],
        right_on=["iso_code", "date_dt"],
        how="left",
        validate="m:1",
    )

    panel = panel.drop(columns=["iso_code", "date_dt"])
    panel = panel.sort_values(["CountryName", "Date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    return panel


def main() -> None:
    output_path = Path("data/oxford/oxford_panel.csv")
    panel = build_panel(output_path)

    n_countries = panel["CountryName"].nunique()
    date_min = str(panel["Date"].min())
    date_max = str(panel["Date"].max())

    missing_cases = float(panel["new_cases_smoothed_per_million"].isna().mean())
    missing_deaths = float(panel["new_deaths_smoothed_per_million"].isna().mean())

    print("\nBuilt Oxford panel dataset")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(panel):,}")
    print(f"  Countries: {n_countries}")
    print(f"  Date range: {date_min} -> {date_max}")
    print(f"  Missing new_cases_smoothed_per_million: {missing_cases:.2%}")
    print(f"  Missing new_deaths_smoothed_per_million: {missing_deaths:.2%}")


if __name__ == "__main__":
    main()
