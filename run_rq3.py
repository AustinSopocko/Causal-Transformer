#!/usr/bin/env python
"""
RQ3: How robust and reliable is the CRT?

Evaluate stability of predictions and policy recommendations under:
1. Input perturbation (a_hist, y_hist noise)
2. Template ranking stability under perturbation
3. Subsampling variation (different test-window seeds)

Usage:
    python run_rq3.py --checkpoint checkpoints/oxford/best_crt.pt --oxford_csv data/oxford/oxford_panel.csv --config src/configs/oxford_config.yaml
    python run_rq3.py ... --experiments perturbation ranking subsampling
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from crt.model import CRTModel
from crt.rollout import rollout
from src.data.normalise import OutcomeScaler, inverse_transform_outcomes
from src.data.panel_windows import PanelWindows

from run_rq1 import build_test_windows, load_checkpoint, load_yaml_config
from run_rq2 import (
    TEMPLATE_NAMES,
    build_template_a_fut,
    run_rq2_phase1,
    run_rq2_phase2,
)

POLICY_CLIP = (0.0, 100.0)


def perturb_inputs(
    windows: PanelWindows,
    rng: np.random.Generator,
    sigma_a: float,
    sigma_y: float,
) -> PanelWindows:
    """Add Gaussian noise to a_hist and y_hist; clip policies to [0, 100]."""
    a_pert = windows.a_hist + sigma_a * torch.from_numpy(
        rng.standard_normal(windows.a_hist.shape).astype(np.float32)
    )
    a_pert = torch.clamp(a_pert, min=POLICY_CLIP[0], max=POLICY_CLIP[1])

    y_pert = windows.y_hist + sigma_y * torch.from_numpy(
        rng.standard_normal(windows.y_hist.shape).astype(np.float32)
    )

    return PanelWindows(
        x_hist=windows.x_hist,
        a_hist=a_pert,
        y_hist=y_pert,
        a_fut=windows.a_fut,
        y_fut=windows.y_fut,
        country_idx=windows.country_idx,
        metadata=windows.metadata,
    )


def run_experiment1_perturbation(
    model: CRTModel,
    test_windows: PanelWindows,
    scaler: Optional[OutcomeScaler],
    config,
    device: str,
    output_dir: Path,
    sigma_list: List[float],
    n_replicates: int,
    batch_size: int = 64,
) -> None:
    """Experiment 1: Input perturbation — MAE and CV of cumulative cases vs baseline."""
    H = config.forecast_horizon
    n_test = len(test_windows)

    # Baseline rollout (status_quo)
    baseline_cases = []
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        x_hist = test_windows.x_hist[start:end].to(device)
        a_hist = test_windows.a_hist[start:end].to(device)
        y_hist = test_windows.y_hist[start:end].to(device)
        a_fut = test_windows.a_fut[start:end].to(device)
        country_idx_batch = test_windows.country_idx[start:end].to(device)
        a_template = build_template_a_fut(a_fut, "status_quo", H)

        with torch.no_grad():
            y_pred = rollout(model, x_hist, a_hist, y_hist, a_template, country_idx=country_idx_batch)

        if scaler is not None:
            y_pred = inverse_transform_outcomes(y_pred.cpu(), scaler)
        else:
            y_pred = y_pred.cpu()

        for i in range(end - start):
            cumulative = float(y_pred[i, :, 0].sum().item())
            baseline_cases.append(max(0.0, cumulative))

    baseline_cases = np.array(baseline_cases)

    records = []
    for sigma in sigma_list:
        sigma_a = sigma_y = sigma
        rng = np.random.default_rng(int(sigma * 1000) + 42)

        replicate_cases = []  # (n_replicates, n_windows)
        for rep in range(n_replicates):
            pert_windows = perturb_inputs(test_windows, rng, sigma_a, sigma_y)
            cases = []
            for start in range(0, n_test, batch_size):
                end = min(start + batch_size, n_test)
                x_hist = pert_windows.x_hist[start:end].to(device)
                a_hist = pert_windows.a_hist[start:end].to(device)
                y_hist = pert_windows.y_hist[start:end].to(device)
                a_fut = pert_windows.a_fut[start:end].to(device)
                country_idx_batch = pert_windows.country_idx[start:end].to(device)
                a_template = build_template_a_fut(a_fut, "status_quo", H)

                with torch.no_grad():
                    y_pred = rollout(
                        model, x_hist, a_hist, y_hist, a_template, country_idx=country_idx_batch
                    )

                if scaler is not None:
                    y_pred = inverse_transform_outcomes(y_pred.cpu(), scaler)
                else:
                    y_pred = y_pred.cpu()

                for i in range(end - start):
                    cumulative = float(y_pred[i, :, 0].sum().item())
                    cases.append(max(0.0, cumulative))

            replicate_cases.append(cases)

        replicate_cases = np.array(replicate_cases)  # (n_replicates, n_windows)
        mae_per_window = np.abs(replicate_cases - baseline_cases).mean(axis=0)
        cv_per_window = np.std(replicate_cases, axis=0) / (np.mean(replicate_cases, axis=0) + 1e-8)

        records.append({
            "sigma_a": sigma_a,
            "sigma_y": sigma_y,
            "mae_mean": float(np.mean(mae_per_window)),
            "mae_std": float(np.std(mae_per_window)),
            "cv_mean": float(np.mean(cv_per_window)),
            "cv_std": float(np.std(cv_per_window)),
            "n_windows": n_test,
            "n_replicates": n_replicates,
        })

    df = pd.DataFrame(records)
    path = output_dir / "rq3_prediction_stability.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}")

    # Same as summary for this experiment
    summary_path = output_dir / "prediction_stability_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


def _run_rq2_phase1_with_windows(
    model: CRTModel,
    windows: PanelWindows,
    scaler: Optional[OutcomeScaler],
    config,
    device: str,
    batch_size: int = 64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run RQ2 Phase 1 logic on given windows. Returns (ranking, by_country)."""
    H = config.forecast_horizon
    n_test = len(windows)
    records: List[Dict] = []

    for template in TEMPLATE_NAMES:
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_hist = windows.x_hist[start:end].to(device)
            a_hist = windows.a_hist[start:end].to(device)
            y_hist = windows.y_hist[start:end].to(device)
            a_fut = windows.a_fut[start:end].to(device)
            country_idx_batch = windows.country_idx[start:end].to(device)
            a_template = build_template_a_fut(a_fut, template, H)

            with torch.no_grad():
                y_pred = rollout(
                    model, x_hist, a_hist, y_hist, a_template, country_idx=country_idx_batch
                )

            if scaler is not None:
                y_pred = inverse_transform_outcomes(y_pred.cpu(), scaler)
            else:
                y_pred = y_pred.cpu()

            for i in range(end - start):
                cumulative_cases = float(y_pred[i, :, 0].sum().item())
                wid = start + i
                country = windows.metadata.iloc[wid]["country"]
                records.append({
                    "window_id": wid,
                    "country": country,
                    "template": template,
                    "cumulative_cases": max(0.0, cumulative_cases),
                })

    df = pd.DataFrame(records)
    ranking = df.groupby("template")["cumulative_cases"].agg(["mean", "std", "count"]).reset_index()
    ranking.columns = ["template", "mean_cumulative_cases", "std", "n_windows"]
    ranking = ranking.sort_values("mean_cumulative_cases").reset_index(drop=True)

    by_country = df.groupby(["country", "template"])["cumulative_cases"].mean().reset_index()
    by_country.columns = ["country", "template", "mean_cases"]

    return ranking, by_country


def run_experiment2_ranking(
    model: CRTModel,
    test_windows: PanelWindows,
    scaler: Optional[OutcomeScaler],
    config,
    device: str,
    output_dir: Path,
    rq2_by_country_path: Optional[Path],
    cluster_assignments_path: Optional[Path],
    best_template_path: Optional[Path],
    sigma: float = 0.05,
    n_replicates: int = 20,
) -> None:
    """Experiment 2: Policy recommendation stability — Kendall tau and best-template agreement."""
    # Baseline
    baseline_ranking, baseline_by_country = _run_rq2_phase1_with_windows(
        model, test_windows, scaler, config, device
    )
    baseline_order = baseline_ranking["template"].tolist()

    # Load cluster assignments and baseline best per cluster
    cluster_assign = None
    baseline_best_per_cluster = None
    if cluster_assignments_path and cluster_assignments_path.exists():
        cluster_assign = pd.read_csv(cluster_assignments_path)
        if best_template_path and best_template_path.exists():
            best_df = pd.read_csv(best_template_path)
            baseline_best_per_cluster = dict(zip(best_df["cluster"], best_df["winning_template"]))
        else:
            # Compute from baseline_by_country (same logic as RQ2 Phase 2)
            merged = baseline_by_country.merge(cluster_assign, on="country", how="inner")
            baseline_best_per_cluster = {}
            for cid in merged["cluster"].unique():
                sub = merged[merged["cluster"] == cid]
                best_t = "status_quo"
                best_mean = float("inf")
                for t in sub["template"].unique():
                    t_mean = sub[sub["template"] == t]["mean_cases"].mean()
                    if t_mean < best_mean:
                        best_mean = t_mean
                        best_t = t
                baseline_best_per_cluster[int(cid)] = best_t

    rng = np.random.default_rng(42)
    kendall_taus = []
    ranking_records = []
    replicate_by_countries: List[pd.DataFrame] = []

    for rep in range(n_replicates):
        pert_windows = perturb_inputs(test_windows, rng, sigma, sigma)
        pert_ranking, pert_by_country = _run_rq2_phase1_with_windows(
            model, pert_windows, scaler, config, device
        )
        replicate_by_countries.append(pert_by_country)
        pert_order = pert_ranking["template"].tolist()

        # Kendall tau on rankings (by mean_cumulative_cases order)
        rank_baseline = {t: i for i, t in enumerate(baseline_order)}
        rank_pert = {t: i for i, t in enumerate(pert_order)}
        x = [rank_baseline[t] for t in TEMPLATE_NAMES]
        y = [rank_pert[t] for t in TEMPLATE_NAMES]
        tau, _ = kendalltau(x, y)
        kendall_taus.append(tau)

        ranking_records.append({
            "replicate": rep,
            "sigma": sigma,
            "kendall_tau": tau,
            "ranking": " | ".join(pert_order),
        })

    df_rank = pd.DataFrame(ranking_records)
    path_rank = output_dir / "rq3_ranking_per_replicate.csv"
    df_rank.to_csv(path_rank, index=False)
    print(f"Saved {path_rank}")

    stability_record = {
        "sigma": sigma,
        "kendall_tau_mean": float(np.mean(kendall_taus)),
        "kendall_tau_std": float(np.std(kendall_taus)),
        "n_replicates": n_replicates,
    }

    # Best-template agreement per cluster (using already-computed replicate by_countries)
    if cluster_assign is not None and baseline_best_per_cluster is not None:
        agreements = []
        agreement_records = []
        for cid, baseline_winner in baseline_best_per_cluster.items():
            count_agree = 0
            for pert_by_country in replicate_by_countries:
                merged = pert_by_country.merge(cluster_assign, on="country", how="inner")
                sub = merged[merged["cluster"] == cid]
                if sub.empty:
                    continue
                best_t = sub.groupby("template")["mean_cases"].mean().idxmin()
                if best_t == baseline_winner:
                    count_agree += 1
            pct = 100.0 * count_agree / n_replicates if n_replicates > 0 else 0.0
            agreements.append(pct)
            agreement_records.append({
                "cluster": cid,
                "baseline_winner": baseline_winner,
                "agreement_pct": pct,
                "n_replicates": n_replicates,
            })

        stability_record["best_template_agreement_mean"] = float(np.mean(agreements))
        df_agree = pd.DataFrame(agreement_records)
        path_agree = output_dir / "rq3_best_template_agreement.csv"
        df_agree.to_csv(path_agree, index=False)
        print(f"Saved {path_agree}")

    df_stab = pd.DataFrame([stability_record])
    path_stab = output_dir / "rq3_ranking_stability.csv"
    df_stab.to_csv(path_stab, index=False)
    print(f"Saved {path_stab}")


def run_experiment3_subsampling(
    model: CRTModel,
    test_windows: PanelWindows,
    scaler: Optional[OutcomeScaler],
    config,
    device: str,
    output_dir: Path,
    max_windows: int,
    seeds: List[int],
) -> None:
    """Experiment 3: Subsampling stability — Kendall tau of template ranking vs baseline seed."""
    n_test = len(test_windows)
    baseline_seed = seeds[0]

    def get_ranking_for_seed(seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_test, size=min(max_windows, n_test), replace=False)
        sub_windows = test_windows.subset(idx)
        ranking, _ = _run_rq2_phase1_with_windows(model, sub_windows, scaler, config, device)
        return ranking

    baseline_ranking = get_ranking_for_seed(baseline_seed)
    baseline_order = baseline_ranking["template"].tolist()
    rank_baseline = {t: i for i, t in enumerate(baseline_order)}

    records = []
    for seed in seeds:
        if seed == baseline_seed:
            tau = 1.0
        else:
            pert_ranking = get_ranking_for_seed(seed)
            pert_order = pert_ranking["template"].tolist()
            rank_pert = {t: i for i, t in enumerate(pert_order)}
            x = [rank_baseline[t] for t in TEMPLATE_NAMES]
            y = [rank_pert[t] for t in TEMPLATE_NAMES]
            tau, _ = kendalltau(x, y)
        records.append({"seed": seed, "kendall_tau_vs_baseline": tau})

    df = pd.DataFrame(records)
    path = output_dir / "rq3_subsampling_stability.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def run_rq3(
    checkpoint_path: str | Path,
    oxford_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path = "results/rq3",
    experiments: Optional[List[str]] = None,
    max_windows: Optional[int] = 500,
    n_replicates: int = 20,
    sigma_list: Optional[List[float]] = None,
    ranking_sigma: float = 0.05,
    subsampling_seeds: Optional[List[int]] = None,
    rq2_output_dir: Optional[str | Path] = None,
    context_csv: Optional[str | Path] = None,
    n_clusters: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = experiments or ["perturbation", "ranking", "subsampling"]
    sigma_list = sigma_list or [0.01, 0.05, 0.10]
    subsampling_seeds = subsampling_seeds or [42, 43, 44, 45, 46, 47, 48]
    rq2_dir = Path(rq2_output_dir) if rq2_output_dir else Path("results/rq2")

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

    n_test = len(test_windows)
    if n_test == 0:
        raise RuntimeError("No test windows; check data and split.")

    # For experiments 1 and 2: subsample with seed 42 for consistency
    full_test_windows = test_windows
    if max_windows and n_test > max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_test, size=max_windows, replace=False)
        test_windows = test_windows.subset(idx)
        n_test = len(test_windows)
    print(f"Using {n_test} test windows (full pool: {len(full_test_windows)})")

    # Generate RQ2 outputs if needed for experiment 2
    by_country_path = rq2_dir / "rq2_by_country.csv"
    cluster_path = rq2_dir / "cluster_assignments.csv"
    best_path = rq2_dir / "best_template_per_cluster.csv"

    if "ranking" in experiments and (not cluster_path.exists() or not best_path.exists()):
        print("Running RQ2 to produce baseline by_country and cluster outputs...")
        run_rq2_phase1(
            model, test_windows, scaler, config, device, rq2_dir,
            max_windows=None, batch_size=64, subsample_seed=42
        )
        if context_csv and Path(context_csv).exists():
            by_country = pd.read_csv(by_country_path)
            run_rq2_phase2(by_country, Path(context_csv), rq2_dir, n_clusters=n_clusters)

    if "perturbation" in experiments:
        print("Experiment 1: Input perturbation...")
        run_experiment1_perturbation(
            model, test_windows, scaler, config, device, out_dir,
            sigma_list=sigma_list, n_replicates=n_replicates,
        )

    if "ranking" in experiments:
        print("Experiment 2: Ranking stability...")
        run_experiment2_ranking(
            model, test_windows, scaler, config, device, out_dir,
            rq2_by_country_path=by_country_path if by_country_path.exists() else None,
            cluster_assignments_path=cluster_path if cluster_path.exists() else None,
            best_template_path=best_path if best_path.exists() else None,
            sigma=ranking_sigma,
            n_replicates=n_replicates,
        )

    if "subsampling" in experiments:
        print("Experiment 3: Subsampling stability...")
        run_experiment3_subsampling(
            model, full_test_windows, scaler, config, device, out_dir,
            max_windows=max_windows or len(full_test_windows),
            seeds=subsampling_seeds,
        )

    # Summary plot
    try:
        import matplotlib.pyplot as plt
        stab_path = out_dir / "rq3_prediction_stability.csv"
        if stab_path.exists():
            df = pd.read_csv(stab_path)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax = axes[0]
            ax.bar(range(len(df)), df["mae_mean"], yerr=df["mae_std"], capsize=4, alpha=0.8)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([f"{s}" for s in df["sigma_a"]])
            ax.set_xlabel("Noise sigma")
            ax.set_ylabel("MAE (cumulative cases)")
            ax.set_title("Prediction stability: MAE vs baseline")

            ax = axes[1]
            ax.bar(range(len(df)), df["cv_mean"], yerr=df["cv_std"], capsize=4, alpha=0.8)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([f"{s}" for s in df["sigma_a"]])
            ax.set_xlabel("Noise sigma")
            ax.set_ylabel("Coefficient of variation")
            ax.set_title("Prediction stability: CV across replicates")
            plt.suptitle("RQ3: CRT robustness to input perturbation")
            plt.tight_layout()
            plt.savefig(out_dir / "rq3_stability_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_dir / 'rq3_stability_summary.png'}")
    except Exception as e:
        print(f"Could not save summary plot: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3: Robustness and reliability")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oxford/best_crt.pt")
    parser.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/rq3")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["perturbation", "ranking", "subsampling"],
        help="Experiments to run",
    )
    parser.add_argument("--max_windows", type=int, default=500)
    parser.add_argument("--n_replicates", type=int, default=20)
    parser.add_argument("--sigma_list", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--ranking_sigma", type=float, default=0.05)
    parser.add_argument("--rq2_output_dir", type=str, default="results/rq2")
    parser.add_argument("--context_csv", type=str, default=None)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_rq3(
        checkpoint_path=args.checkpoint,
        oxford_csv=args.oxford_csv,
        config_path=args.config,
        output_dir=args.output_dir,
        experiments=args.experiments,
        max_windows=args.max_windows or None,
        n_replicates=args.n_replicates,
        sigma_list=args.sigma_list,
        ranking_sigma=args.ranking_sigma,
        rq2_output_dir=args.rq2_output_dir,
        context_csv=args.context_csv,
        n_clusters=args.n_clusters,
        device=args.device,
    )


if __name__ == "__main__":
    main()
