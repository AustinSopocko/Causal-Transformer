#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from crt.rollout import rollout
from evaluate_oxford_extended import (
    build_raw_train_test_windows,
    compute_ar_ridge_prediction,
    compute_persistence_prediction,
    filter_negligible_countries,
    write_plain_markdown_table,
)
from run_rq1 import load_checkpoint
from src.data.normalise import apply_outcome_scaler, inverse_transform_outcomes
from src.eval.oxford_extended import compute_window_segment_rmse


def getPlt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def loadJson(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def buildConstantPolicy(aFutObs: torch.Tensor, policyCols: List[str], scope: str, stringencyCol: str) -> torch.Tensor:
    out = aFutObs.clone()
    if scope == "all_policies":
        out[:] = aFutObs[:, :1, :]
        return out
    if scope == "stringency_only":
        if stringencyCol not in policyCols:
            raise ValueError("Missing stringency column")
        j = int(policyCols.index(stringencyCol))
        out[:, :, j] = aFutObs[:, :1, j]
        return out
    raise ValueError("Unknown constant scope")


def buildShuffledPolicy(aFutObs: torch.Tensor, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    n = int(aFutObs.shape[0])
    perm = rng.permutation(n)
    idx = torch.from_numpy(perm).to(dtype=torch.long)
    return aFutObs[idx].clone()


def predictWithPolicy(
    model,
    scaler,
    testRaw,
    aFut,
    log1pOutcomes: bool,
    device: str,
    batchSize: int,
    clipNonneg: bool,
) -> torch.Tensor:
    model.eval()
    testScaled = apply_outcome_scaler(testRaw, scaler, log1p=log1pOutcomes)
    n = len(testRaw)
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, int(batchSize)):
            end = min(start + int(batchSize), n)
            xHist = testScaled.x_hist[start:end].to(device)
            aHist = testScaled.a_hist[start:end].to(device)
            yHist = testScaled.y_hist[start:end].to(device)
            aFutB = aFut[start:end].to(device)
            cIdx = testScaled.country_idx[start:end].to(device)
            yScaled = rollout(model=model, x_hist=xHist, a_hist=aHist, y_hist=yHist, a_fut=aFutB, country_idx=cIdx)
            yRaw = inverse_transform_outcomes(yScaled.cpu(), scaler)
            out.append(yRaw)
    yPred = torch.cat(out, dim=0)
    if clipNonneg:
        yPred = torch.clamp(yPred, min=0.0)
    return yPred


def computeBaseRMSE(windowRmse: np.ndarray) -> float:
    if windowRmse.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(windowRmse * windowRmse)))


def summarizeScenario(name: str, yPred: torch.Tensor, yTrue: torch.Tensor, lateStart: int, policyMask: np.ndarray) -> Dict[str, float | str]:
    h = int(yTrue.shape[1])
    allRmse = compute_window_segment_rmse(yPred, yTrue, start_1b=1, end_1b=h)
    lateRmse = compute_window_segment_rmse(yPred, yTrue, start_1b=lateStart, end_1b=h)
    subLate = lateRmse[policyMask] if int(np.sum(policyMask)) > 0 else np.array([], dtype=np.float64)
    return {
        "scenario": name,
        "overall_rmse": computeBaseRMSE(allRmse),
        "late_horizon_rmse": computeBaseRMSE(lateRmse),
        "policy_subset_late_rmse": computeBaseRMSE(subLate),
        "policy_subset_n": int(np.sum(policyMask)),
    }


def saveLateRmsePlot(df: pd.DataFrame, outPath: Path) -> None:
    plt = getPlt()
    d = df.copy()
    order = ["observed_policy", "constant_policy", "shuffled_policy"]
    d["ord"] = d["scenario"].map({k: i for i, k in enumerate(order)}).fillna(999)
    d = d.sort_values("ord")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(d))
    ax.bar(x, d["late_horizon_rmse"].to_numpy(dtype=float), color=["#2563eb", "#dc2626", "#16a34a"][: len(d)])
    ax.set_xticks(x)
    ax.set_xticklabels(d["scenario"].tolist())
    ax.set_ylabel("Late RMSE (days 32-42)")
    ax.set_title("Synthetic policy sensitivity")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(d["late_horizon_rmse"].to_numpy(dtype=float)):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(outPath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default="results/stage4_h35_week46_h2h_with_baselines")
    p.add_argument("--checkpoint", type=str, default="checkpoints/oxford_stage4_h35_week46/03_huber_anchor02_nonneg/best_crt.pt")
    p.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    p.add_argument("--config", type=str, default="src/configs/stage4_h35_week46/03_huber_anchor02_nonneg.yaml")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--constant_scope", choices=["all_policies", "stringency_only"], default="all_policies")
    p.add_argument("--stringency_col", type=str, default="StringencyIndex_Average")
    p.add_argument("--policy_subset_csv", type=str, default="")
    p.add_argument("--observed_prt_cache", type=str, default="pred_cache/c03.npz")
    p.add_argument("--out_dir", type=str, default="")
    a = p.parse_args()

    evalDir = Path(a.eval_dir)
    outDir = Path(a.out_dir) if a.out_dir else evalDir / "synthetic_policy_sensitivity"
    outDir.mkdir(parents=True, exist_ok=True)

    meta = loadJson(evalDir / "evaluation_metadata.json")
    policyCols = list(meta.get("policy_cols", []))
    outcomeCols = list(meta.get("outcome_cols", []))
    clipNonneg = bool(meta.get("clip_nonnegative", False))
    lateStart = int(meta.get("late_horizon_start", 32))

    model, _, scaler, countryToIdx, ckptPolicy, ckptOutcome, stateCols = load_checkpoint(a.checkpoint, device=a.device)
    if scaler is None:
        raise RuntimeError("Missing scaler")
    if ckptPolicy:
        policyCols = list(ckptPolicy)
    if ckptOutcome:
        outcomeCols = list(ckptOutcome)
    stateCols = list(stateCols or [])

    trainRaw, testRaw, log1pOutcomes = build_raw_train_test_windows(
        oxford_csv=a.oxford_csv,
        config_path=a.config,
        policy_cols=policyCols,
        outcome_cols=outcomeCols,
        state_cols=stateCols,
        country_to_idx=countryToIdx,
    )

    filt = meta.get("country_filter", {})
    droppedCountries: List[str] = []
    if bool(filt.get("enabled", False)):
        trainRaw, testRaw, countrySummary, droppedCountries = filter_negligible_countries(
            train_raw=trainRaw,
            test_raw=testRaw,
            outcome_cols=outcomeCols,
            outcome_name=str(filt.get("outcome", "new_cases_smoothed_per_million")),
            history_len=int(filt.get("history_len", 21)),
            threshold=float(filt.get("threshold", 5.0)),
        )
        countrySummary.to_csv(outDir / "country_exclusion_summary.csv", index=False)
        (outDir / "countries_dropped_negligible.txt").write_text(
            "\n".join(sorted(droppedCountries)) + ("\n" if droppedCountries else ""),
            encoding="utf-8",
        )

    n = len(testRaw)
    h = int(testRaw.y_fut.shape[1])
    lateStart = max(1, min(lateStart, h))

    subsetPath = Path(a.policy_subset_csv) if a.policy_subset_csv else evalDir / "policy_change_subset_windows.csv"
    subsetDf = pd.read_csv(subsetPath).sort_values("window_index").reset_index(drop=True)
    idx = np.arange(n, dtype=np.int64)
    if subsetDf.shape[0] != n or not np.array_equal(subsetDf["window_index"].to_numpy(dtype=np.int64), idx):
        raise ValueError("Window mismatch")
    policyMask = subsetDf["is_policy_change_subset"].to_numpy(dtype=bool)

    aObs = testRaw.a_fut.clone()
    aConst = buildConstantPolicy(aFutObs=aObs, policyCols=policyCols, scope=str(a.constant_scope), stringencyCol=str(a.stringency_col))
    aShuf = buildShuffledPolicy(aFutObs=aObs, seed=int(a.seed))

    yTrue = testRaw.y_fut
    cachePath = Path(a.observed_prt_cache)
    if not cachePath.is_absolute():
        cachePath = evalDir / cachePath

    if cachePath.exists():
        yObsNp = np.load(cachePath, allow_pickle=True)["y_pred"]
        yObs = torch.from_numpy(yObsNp.astype(np.float32))
        if clipNonneg:
            yObs = torch.clamp(yObs, min=0.0)
    else:
        yObs = predictWithPolicy(
            model=model,
            scaler=scaler,
            testRaw=testRaw,
            aFut=aObs,
            log1pOutcomes=log1pOutcomes,
            device=a.device,
            batchSize=int(a.batch_size),
            clipNonneg=clipNonneg,
        )

    yConst = predictWithPolicy(
        model=model,
        scaler=scaler,
        testRaw=testRaw,
        aFut=aConst,
        log1pOutcomes=log1pOutcomes,
        device=a.device,
        batchSize=int(a.batch_size),
        clipNonneg=clipNonneg,
    )
    yShuf = predictWithPolicy(
        model=model,
        scaler=scaler,
        testRaw=testRaw,
        aFut=aShuf,
        log1pOutcomes=log1pOutcomes,
        device=a.device,
        batchSize=int(a.batch_size),
        clipNonneg=clipNonneg,
    )

    yBase = compute_persistence_prediction(testRaw)
    yAr = compute_ar_ridge_prediction(train_raw=trainRaw, test_raw=testRaw, lag_len=21, alpha=1.0, include_country_context=False)
    if clipNonneg:
        yBase = torch.clamp(yBase, min=0.0)
        yAr = torch.clamp(yAr, min=0.0)

    rows = [
        summarizeScenario("observed_policy", yObs, yTrue, lateStart=lateStart, policyMask=policyMask),
        summarizeScenario("constant_policy", yConst, yTrue, lateStart=lateStart, policyMask=policyMask),
        summarizeScenario("shuffled_policy", yShuf, yTrue, lateStart=lateStart, policyMask=policyMask),
        summarizeScenario("persistence_reference", yBase, yTrue, lateStart=lateStart, policyMask=policyMask),
        summarizeScenario("ar_ridge_reference", yAr, yTrue, lateStart=lateStart, policyMask=policyMask),
    ]
    tableDf = pd.DataFrame(rows)

    obsLate = float(tableDf.loc[tableDf["scenario"] == "observed_policy", "late_horizon_rmse"].iloc[0])
    obsSub = float(tableDf.loc[tableDf["scenario"] == "observed_policy", "policy_subset_late_rmse"].iloc[0])
    tableDf["delta_late_rmse_vs_observed_policy"] = tableDf["late_horizon_rmse"] - obsLate
    tableDf["delta_policy_subset_late_rmse_vs_observed_policy"] = tableDf["policy_subset_late_rmse"] - obsSub

    predDir = outDir / "pred_cache"
    predDir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predDir / "observed_policy.npz", y_pred=yObs.numpy())
    np.savez_compressed(predDir / "constant_policy.npz", y_pred=yConst.numpy())
    np.savez_compressed(predDir / "shuffled_policy.npz", y_pred=yShuf.numpy())

    tableCsv = outDir / "table_synthetic_policy_effect.csv"
    tableMd = outDir / "table_synthetic_policy_effect.md"
    tableDf.to_csv(tableCsv, index=False)
    write_plain_markdown_table(tableDf, tableMd, precision=4)

    plotDf = tableDf[tableDf["scenario"].isin(["observed_policy", "constant_policy", "shuffled_policy"])].copy()
    figPath = outDir / "fig_synthetic_policy_late_rmse.png"
    saveLateRmsePlot(plotDf, figPath)

    outMeta = {
        "eval_dir": str(evalDir),
        "checkpoint": str(a.checkpoint),
        "oxford_csv": str(a.oxford_csv),
        "config": str(a.config),
        "device": str(a.device),
        "n_windows": int(n),
        "horizon": int(h),
        "late_start": int(lateStart),
        "policy_subset_n": int(np.sum(policyMask)),
        "policy_subset_ratio": float(np.mean(policyMask)),
        "constant_scope": str(a.constant_scope),
        "stringency_col": str(a.stringency_col),
        "clip_nonnegative": bool(clipNonneg),
        "countries_dropped_n": int(len(droppedCountries)),
    }
    with open(outDir / "synthetic_policy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(outMeta, f, indent=2)

    print("done")
    print(str(outDir))


if __name__ == "__main__":
    main()

