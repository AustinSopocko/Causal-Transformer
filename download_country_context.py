#!/usr/bin/env python
"""
Download country context from OWID and build country_context.csv for RQ2 clustering.

Columns: country, CountryCode, population_density, median_age, gdp_per_capita, hospital_beds_per_thousand

Usage:
    python download_country_context.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

CONTEXT_COLUMNS = [
    "iso_code",
    "location",
    "population_density",
    "median_age",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
]


def build_country_context(output_path: Path) -> pd.DataFrame:
    print("Downloading OWID data...")
    df = pd.read_csv(OWID_URL, usecols=[c for c in CONTEXT_COLUMNS if c != "location"] + ["location"])

    # Keep country-level (iso_code length 3)
    df = df[df["iso_code"].str.len() == 3].copy()
    df = df.dropna(subset=["iso_code"])

    # For each country: median over time per context column
    agg_cols = ["population_density", "median_age", "gdp_per_capita", "hospital_beds_per_thousand"]
    if "aged_65_older" in df.columns:
        agg_cols.append("aged_65_older")

    available = [c for c in agg_cols if c in df.columns]
    if not available:
        raise RuntimeError("No context columns found in OWID data.")

    agg_dict = {"location": "first"}
    for c in available:
        agg_dict[c] = "median"
    ctx = df.groupby("iso_code").agg(agg_dict).reset_index()

    # Rename for consistency
    ctx = ctx.rename(columns={"iso_code": "CountryCode", "location": "country"})
    if "aged_65_older" in ctx.columns and "median_age" not in ctx.columns:
        ctx = ctx.rename(columns={"aged_65_older": "age65"})
    elif "aged_65_older" in ctx.columns:
        ctx["age65"] = ctx["aged_65_older"]

    # Drop rows with >50% missing
    required = ["population_density", "median_age", "gdp_per_capita", "hospital_beds_per_thousand"]
    present = [c for c in required if c in ctx.columns]
    ctx = ctx.dropna(subset=present, thresh=len(present) // 2)

    # Impute remaining NaNs with median
    for c in present:
        ctx[c] = ctx[c].fillna(ctx[c].median())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.to_csv(output_path, index=False)
    return ctx


def main() -> None:
    output_path = Path("data/oxford/country_context.csv")
    ctx = build_country_context(output_path)
    print(f"Saved {output_path}")
    print(f"Countries: {len(ctx)}")
    print(ctx[["country", "CountryCode"] + [c for c in ctx.columns if c not in ("country", "CountryCode")]].head(10))


if __name__ == "__main__":
    main()
