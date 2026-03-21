#!/usr/bin/env python
"""
RQ2: Policy templates that work (and for whom).

Phase 1: Define templates, rollout, aggregate by template and country.
Phase 2: If context_csv provided, cluster countries and compute best template per cluster with bootstrap CI.

Usage:
    python run_rq2.py --checkpoint checkpoints/oxford/best_crt.pt --oxford_csv data/oxford/oxford_panel.csv --config src/configs/oxford_config.yaml
    python run_rq2.py ... --context_csv data/oxford/country_context.csv --n_clusters 4

If Phase 2 segfaults (PyTorch + sklearn), run with: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_rq2.py ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from crt.model import CRTModel
from crt.rollout import rollout
from src.data.normalise import OutcomeScaler, inverse_transform_outcomes

from run_rq1 import build_test_windows, load_checkpoint, load_yaml_config

TEMPLATE_NAMES = ["status_quo", "early_hard", "gradual", "reactive_pulse"]
POLICY_CLIP = (0.0, 100.0)


def build_template_a_fut(a_fut: torch.Tensor, template_name: str, H: int) -> torch.Tensor:
    """Build alternative a_fut for a named policy template. Policies clipped to [0, 100]."""
    B, _, d_a = a_fut.shape
    if H is None or H <= 0:
        H = a_fut.shape[1]

    if template_name == "status_quo":
        return a_fut.clone()

    obs_mean = a_fut.mean(dim=1, keepdim=True)
    alt = a_fut.clone()

    if template_name == "early_hard":
        # Tighten immediately at high intensity
        alt = a_fut * 1.2

    elif template_name == "gradual":
        # Linear ramp from 0.5× mean to 1.0× observed over H weeks
        low = obs_mean * 0.5
        for h in range(min(H, alt.shape[1])):
            t = (h + 1) / H
            alt[:, h : h + 1, :] = low * (1 - t) + a_fut[:, h : h + 1, :] * t

    elif template_name == "reactive_pulse":
        # First H/2 at 0.6×, second H/2 at 1.0× observed
        mid = H // 2
        if mid > 0:
            alt[:, :mid, :] = a_fut[:, :mid, :] * 0.6
        alt[:, mid:, :] = a_fut[:, mid:, :] * 1.0

    else:
        raise ValueError(f"Unknown template: {template_name}")

    # Clip to valid policy range
    alt = torch.clamp(alt, min=POLICY_CLIP[0], max=POLICY_CLIP[1])
    return alt


def run_rq2_phase1(
    model: CRTModel,
    test_windows,
    scaler: OutcomeScaler,
    config,
    device: str,
    output_dir: Path,
    max_windows: Optional[int] = 1000,
    batch_size: int = 64,
    subsample_seed: int = 42,
) -> pd.DataFrame:
    """Roll out each template, aggregate by template and by country. Returns rq2_by_country DataFrame."""
    n_test = len(test_windows)
    if max_windows and n_test > max_windows:
        rng = np.random.default_rng(subsample_seed)
        idx = rng.choice(n_test, size=max_windows, replace=False)
        test_windows = test_windows.subset(idx)
        n_test = len(test_windows)
    print(f"Using {n_test} test windows")

    H = config.forecast_horizon
    records: List[Dict] = []

    for template in TEMPLATE_NAMES:
        print(f"  template={template}")
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_hist = test_windows.x_hist[start:end].to(device)
            a_hist = test_windows.a_hist[start:end].to(device)
            y_hist = test_windows.y_hist[start:end].to(device)
            a_fut = test_windows.a_fut[start:end].to(device)
            country_idx_batch = test_windows.country_idx[start:end].to(device)

            a_template = build_template_a_fut(a_fut, template, H)

            with torch.no_grad():
                y_pred = rollout(
                    model, x_hist, a_hist, y_hist, a_template,
                    country_idx=country_idx_batch,
                )

            if scaler is not None:
                y_pred = inverse_transform_outcomes(y_pred.cpu(), scaler)
            else:
                y_pred = y_pred.cpu()

            for i in range(end - start):
                cumulative_cases = float(y_pred[i, :, 0].sum().item())
                wid = start + i
                country = test_windows.metadata.iloc[wid]["country"]
                records.append({
                    "window_id": wid,
                    "country": country,
                    "template": template,
                    "cumulative_cases": cumulative_cases,
                })

    df = pd.DataFrame(records)
    df["cumulative_cases"] = df["cumulative_cases"].clip(lower=0)

    # Template ranking (overall)
    ranking = df.groupby("template")["cumulative_cases"].agg(["mean", "std", "count"]).reset_index()
    ranking.columns = ["template", "mean_cumulative_cases", "std", "n_windows"]
    ranking = ranking.sort_values("mean_cumulative_cases")
    ranking_path = output_dir / "rq2_template_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"Saved {ranking_path}")

    # By country (for Phase 2)
    by_country = df.groupby(["country", "template"])["cumulative_cases"].mean().reset_index()
    by_country.columns = ["country", "template", "mean_cases"]
    by_country_path = output_dir / "rq2_by_country.csv"
    by_country.to_csv(by_country_path, index=False)
    print(f"Saved {by_country_path}")

    # Bar chart: normalise to % of status_quo so differences are visible
    try:
        import matplotlib.pyplot as plt
        baseline_mean = ranking[ranking["template"] == "status_quo"]["mean_cumulative_cases"].iloc[0]
        ranking["pct_of_baseline"] = 100 * ranking["mean_cumulative_cases"] / baseline_mean
        ranking["pct_std"] = 100 * ranking["std"] / baseline_mean

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(ranking))
        ax.bar(x, ranking["pct_of_baseline"], yerr=ranking["pct_std"], capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ranking["template"], rotation=15, ha="right")
        ax.set_ylabel("% of status quo (predicted cumulative cases)")
        ax.set_title("RQ2: Policy template comparison — % of status quo burden")
        ax.axhline(100, color="gray", linestyle="--", alpha=0.7, linewidth=1)
        # Zoom y-axis to emphasise differences (97–103% or data range ± margin)
        lo = max(97, ranking["pct_of_baseline"].min() - 0.5)
        hi = min(103, ranking["pct_of_baseline"].max() + 0.5)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        chart_path = output_dir / "template_bar_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {chart_path}")
    except Exception as e:
        print(f"Could not save bar chart: {e}")

    return by_country


def run_rq2_phase2(
    by_country: pd.DataFrame,
    context_path: Path,
    output_dir: Path,
    n_clusters: int = 4,
    n_bootstrap: int = 1000,
) -> None:
    """Cluster countries, compute best template per cluster with bootstrap CI."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not found; skipping Phase 2 clustering.")
        return

    ctx = pd.read_csv(context_path)
    # Expect: country or CountryCode, density, median_age (or age65), gdp_per_capita, hospital_beds_per_thousand
    country_col = "country" if "country" in ctx.columns else "CountryCode"
    # Use one of: population_density/density, one of: median_age/age65/aged_65_older, gdp, beds
    feature_cols = []
    for c in ["population_density", "density"]:
        if c in ctx.columns:
            feature_cols.append(c)
            break
    for c in ["median_age", "age65", "aged_65_older"]:
        if c in ctx.columns:
            feature_cols.append(c)
            break
    for c in ["gdp_per_capita", "gdp"]:
        if c in ctx.columns:
            feature_cols.append(c)
            break
    for c in ["hospital_beds_per_thousand", "hospital_beds"]:
        if c in ctx.columns:
            feature_cols.append(c)
            break
    if len(feature_cols) < 2:
        # Try common OWID names
        for c in ctx.columns:
            if c.lower() in ("population_density", "median_age", "gdp_per_capita", "hospital_beds_per_thousand"):
                feature_cols.append(c)
    if not feature_cols:
        print("No context features found; skipping Phase 2.")
        return

    # One row per country: take latest/mode
    ctx_agg = ctx.groupby(country_col)[feature_cols].agg(lambda x: x.dropna().median()).reset_index()
    ctx_agg = ctx_agg.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)
    # Impute remaining NaNs with median
    for c in feature_cols:
        ctx_agg[c] = ctx_agg[c].fillna(ctx_agg[c].median())

    X = ctx_agg[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ctx_agg["cluster"] = kmeans.fit_predict(X_scaled)
    cluster_assign = ctx_agg[[country_col, "cluster"]].rename(columns={country_col: "country"})
    cluster_path = output_dir / "cluster_assignments.csv"
    cluster_assign.to_csv(cluster_path, index=False)
    print(f"Saved {cluster_path}")

    # Cluster summary
    summary = ctx_agg.groupby("cluster")[feature_cols].median()
    summary["n_countries"] = ctx_agg.groupby("cluster").size()
    summary_path = output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # Merge by_country with cluster
    merged = by_country.merge(cluster_assign, on="country", how="inner")
    if merged.empty:
        print("No overlap between by_country and context; check country column names.")
        return

    # Best template per cluster with bootstrap CI
    baseline = "status_quo"
    results = []
    for cluster_id in range(n_clusters):
        sub = merged[merged["cluster"] == cluster_id]
        templates = sub["template"].unique()
        baseline_mean = sub[sub["template"] == baseline]["mean_cases"].mean()
        best_template = None
        best_effect = np.inf
        effects = []
        for t in templates:
            if t == baseline:
                continue
            t_mean = sub[sub["template"] == t]["mean_cases"].mean()
            effect = t_mean - baseline_mean  # lower is better
            if effect < best_effect:
                best_effect = effect
                best_template = t
        if best_template is None:
            best_template = baseline
            best_effect = 0.0

        # Bootstrap CI on effect
        rng = np.random.default_rng(42)
        boots = []
        countries = sub["country"].unique()
        for _ in range(n_bootstrap):
            samp = rng.choice(countries, size=len(countries), replace=True)
            sub_boot = sub[sub["country"].isin(samp)]
            b_base = sub_boot[sub_boot["template"] == baseline]["mean_cases"].mean()
            b_best = sub_boot[sub_boot["template"] == best_template]["mean_cases"].mean()
            boots.append(b_best - b_base)
        ci_lo = np.percentile(boots, 2.5)
        ci_hi = np.percentile(boots, 97.5)
        results.append({
            "cluster": cluster_id,
            "winning_template": best_template,
            "mean_effect_vs_baseline": best_effect,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })
    best_df = pd.DataFrame(results)
    best_path = output_dir / "best_template_per_cluster.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Saved {best_path}")


def run_rq2(
    checkpoint_path: str | Path,
    oxford_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path = "results/rq2",
    context_csv: Optional[str | Path] = None,
    n_clusters: int = 4,
    max_windows: Optional[int] = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    model, config, scaler, country_to_idx, policy_cols, outcome_cols, state_cols = load_checkpoint(
        checkpoint_path, device=device
    )
    if not policy_cols or not outcome_cols:
        cfg = load_yaml_config(config_path)
        policy_cols = list(cfg["dataset"]["policy_cols"])
        outcome_cols = list(cfg["dataset"]["outcome_cols"])
        state_cols = list(cfg["dataset"].get("state_cols", []))

    print("Building test windows...")
    test_windows, scaler = build_test_windows(
        oxford_csv, config_path, policy_cols, outcome_cols, state_cols, country_to_idx, scaler=scaler
    )
    if len(test_windows) == 0:
        raise RuntimeError("No test windows; check data and split.")

    print("Phase 1: Rolling out policy templates...")
    by_country = run_rq2_phase1(
        model, test_windows, scaler, config, device, out_dir,
        max_windows=max_windows,
    )

    if context_csv and Path(context_csv).exists():
        print("Phase 2: Clustering and best template per cluster...")
        run_rq2_phase2(by_country, Path(context_csv), out_dir, n_clusters=n_clusters)
    else:
        print("Phase 2 skipped (no context_csv or file not found).")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2: Policy templates")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oxford/best_crt.pt")
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/rq2")
    parser.add_argument("--context_csv", type=str, default=None, help="Country context CSV for Phase 2")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--max_windows", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_rq2(
        checkpoint_path=args.checkpoint,
        oxford_csv=args.oxford_csv,
        config_path=args.config,
        output_dir=args.output_dir,
        context_csv=args.context_csv,
        n_clusters=args.n_clusters,
        max_windows=args.max_windows or None,
        device=args.device,
    )


if __name__ == "__main__":
    main()
