#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from evaluate_oxford_extended import (
    build_raw_train_test_windows,
    compute_ar_ridge_prediction,
    compute_persistence_prediction,
    filter_negligible_countries,
)
from src.eval.oxford_extended import compute_window_segment_rmse


def loadJson(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def saveMdTable(df: pd.DataFrame, outPath: Path, precision: int = 4) -> None:
    outPath.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows: List[str] = []
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{float(v):.{precision}f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    outPath.write_text("\n".join([header, sep] + rows) + "\n", encoding="utf-8")


def weightedMean(x: np.ndarray, w: np.ndarray) -> float:
    d = float(np.sum(w))
    if d <= 0.0:
        return float("nan")
    return float(np.sum(w * x) / d)


def weightedVar(x: np.ndarray, w: np.ndarray) -> float:
    mu = weightedMean(x, w)
    if np.isnan(mu):
        return float("nan")
    d = float(np.sum(w))
    if d <= 0.0:
        return float("nan")
    return float(np.sum(w * (x - mu) ** 2) / d)


def computeSMD(mt: float, vt: float, mc: float, vc: float) -> float:
    p = 0.5 * (vt + vc)
    if p <= 0.0:
        return float("nan")
    return float((mt - mc) / np.sqrt(p))


def buildCovars(yHist: np.ndarray, outcomeIdx: int, slopeLen: int = 14, varLen: int = 21) -> pd.DataFrame:
    y = yHist[:, :, outcomeIdx].astype(np.float64)
    n, t = y.shape
    kSlope = int(max(3, min(slopeLen, t)))
    kVar = int(max(2, min(varLen, t)))

    level = y[:, -1]
    vRecent = np.var(y[:, -kVar:], axis=1)

    yRecent = y[:, -kSlope:]
    x = np.arange(kSlope, dtype=np.float64)
    x0 = x - np.mean(x)
    den = np.sum(x0 * x0)
    if den <= 0.0:
        slope = np.zeros(n, dtype=np.float64)
    else:
        y0 = yRecent - np.mean(yRecent, axis=1, keepdims=True)
        slope = np.sum(y0 * x0.reshape(1, -1), axis=1) / den

    return pd.DataFrame(
        {
            "case_level_t0": level,
            "case_slope_recent": slope,
            "case_variance_recent": vRecent,
            "log1p_case_level_t0": np.log1p(np.clip(level, 0.0, None)),
            "log1p_case_variance_recent": np.log1p(np.clip(vRecent, 0.0, None)),
        }
    )


def fitPropensity(x: np.ndarray, t: np.ndarray, modelKind: str, seed: int) -> Tuple[np.ndarray, str]:
    modelKind = str(modelKind).lower()
    tInt = t.astype(int)

    if modelKind == "gbt":
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

            m = GradientBoostingClassifier(random_state=int(seed))
            m.fit(x, tInt)
            p = m.predict_proba(x)[:, 1].astype(np.float64)
            return p, "gbt"
        except Exception:
            modelKind = "logistic"

    if modelKind != "logistic":
        raise ValueError(f"Unknown model: {modelKind}")

    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    m = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=int(seed))),
        ]
    )
    m.fit(x, tInt)
    p = m.predict_proba(x)[:, 1].astype(np.float64)
    return p, "logistic"


def buildIPW(t: np.ndarray, pHat: np.ndarray, eps: float, qLo: float, qHi: float) -> Dict[str, np.ndarray | float]:
    tF = t.astype(np.float64)
    p = np.clip(pHat.astype(np.float64), eps, 1.0 - eps)
    wRaw = tF / p + (1.0 - tF) / (1.0 - p)
    lo = float(np.quantile(wRaw, qLo))
    hi = float(np.quantile(wRaw, qHi))
    wClip = np.clip(wRaw, lo, hi)
    w = wClip / np.mean(wClip)
    return {
        "pHat": p,
        "wRaw": wRaw,
        "wClip": wClip,
        "w": w,
        "clipLo": lo,
        "clipHi": hi,
    }


def computeBaseRMSE(mse: np.ndarray, weights: np.ndarray | None = None) -> float:
    if mse.size == 0:
        return float("nan")
    if weights is None:
        return float(np.sqrt(np.mean(mse)))
    w = weights.astype(np.float64)
    d = float(np.sum(w))
    if d <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(w * mse) / d))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default="results/stage4_h35_week46_h2h_with_baselines")
    p.add_argument("--oxford_csv", type=str, default="data/oxford/oxford_panel.csv")
    p.add_argument("--config", type=str, default="src/configs/stage4_h35_week46/03_huber_anchor02_nonneg.yaml")
    p.add_argument("--policy_subset_csv", type=str, default="")
    p.add_argument("--prt_pred_cache", type=str, default="pred_cache/c03.npz")
    p.add_argument("--prt_label", type=str, default="c03")
    p.add_argument("--late_start", type=int, default=-1)
    p.add_argument("--ar_lag_len", type=int, default=21)
    p.add_argument("--ar_ridge_alpha", type=float, default=1.0)
    p.add_argument("--ar_include_country_context", action="store_true")
    p.add_argument("--propensity_model", choices=["logistic", "gbt"], default="logistic")
    p.add_argument("--propensity_clip_eps", type=float, default=0.01)
    p.add_argument("--weight_clip_low_q", type=float, default=0.10)
    p.add_argument("--weight_clip_high_q", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="")
    a = p.parse_args()

    evalDir = Path(a.eval_dir)
    outDir = Path(a.out_dir) if a.out_dir else evalDir / "ipw_reweighted"
    outDir.mkdir(parents=True, exist_ok=True)

    meta = loadJson(evalDir / "evaluation_metadata.json")
    policyCols = list(meta.get("policy_cols", []))
    outcomeCols = list(meta.get("outcome_cols", []))
    if "new_cases_smoothed_per_million" not in outcomeCols:
        raise ValueError("Missing cases outcome")
    caseIdx = int(outcomeCols.index("new_cases_smoothed_per_million"))

    trainRaw, testRaw, _ = build_raw_train_test_windows(
        oxford_csv=a.oxford_csv,
        config_path=a.config,
        policy_cols=policyCols,
        outcome_cols=outcomeCols,
        state_cols=[],
        country_to_idx=None,
    )

    c = meta.get("country_filter", {})
    dropped = []
    if bool(c.get("enabled", False)):
        trainRaw, testRaw, _, dropped = filter_negligible_countries(
            train_raw=trainRaw,
            test_raw=testRaw,
            outcome_cols=outcomeCols,
            outcome_name=str(c.get("outcome", "new_cases_smoothed_per_million")),
            history_len=int(c.get("history_len", 21)),
            threshold=float(c.get("threshold", 5.0)),
        )

    n = len(testRaw)
    h = int(testRaw.y_fut.shape[1])
    subsetPath = Path(a.policy_subset_csv) if a.policy_subset_csv else evalDir / "policy_change_subset_windows.csv"
    subsetDf = pd.read_csv(subsetPath).sort_values("window_index").reset_index(drop=True)
    idx = np.arange(n, dtype=np.int64)
    if subsetDf.shape[0] != n or not np.array_equal(subsetDf["window_index"].to_numpy(dtype=np.int64), idx):
        raise ValueError("Window mismatch")
    t = subsetDf["is_policy_change_subset"].to_numpy(dtype=np.int64)

    yHist = testRaw.y_hist.detach().cpu().numpy()
    covDf = buildCovars(yHist=yHist, outcomeIdx=caseIdx, slopeLen=14, varLen=21)
    xCols = ["log1p_case_level_t0", "case_slope_recent", "log1p_case_variance_recent"]
    x = covDf[xCols].to_numpy(dtype=np.float64)

    pHatRaw, fitName = fitPropensity(x=x, t=t, modelKind=a.propensity_model, seed=a.seed)
    ipw = buildIPW(
        t=t,
        pHat=pHatRaw,
        eps=float(a.propensity_clip_eps),
        qLo=float(a.weight_clip_low_q),
        qHi=float(a.weight_clip_high_q),
    )
    w = np.asarray(ipw["w"], dtype=np.float64)
    mask = t.astype(bool)

    clipNonneg = bool(meta.get("clip_nonnegative", False))
    yTrue = testRaw.y_fut

    cachePath = Path(a.prt_pred_cache)
    if not cachePath.is_absolute():
        cachePath = evalDir / cachePath
    pNp = np.load(cachePath, allow_pickle=True)["y_pred"]
    yPrt = torch.from_numpy(pNp.astype(np.float32))
    yAr = compute_ar_ridge_prediction(
        train_raw=trainRaw,
        test_raw=testRaw,
        lag_len=int(a.ar_lag_len),
        alpha=float(a.ar_ridge_alpha),
        include_country_context=bool(a.ar_include_country_context),
    )
    yBase = compute_persistence_prediction(testRaw)

    if clipNonneg:
        yPrt = torch.clamp(yPrt, min=0.0)
        yAr = torch.clamp(yAr, min=0.0)
        yBase = torch.clamp(yBase, min=0.0)

    lateStart = int(a.late_start) if int(a.late_start) > 0 else int(meta.get("late_horizon_start", 32))
    lateStart = max(1, min(lateStart, h))

    preds = {str(a.prt_label): yPrt, "ar_ridge_lag": yAr, "persistence": yBase}

    rows: List[Dict[str, float | str]] = []
    for name, yPred in preds.items():
        lateRmse = compute_window_segment_rmse(yPred, yTrue, start_1b=lateStart, end_1b=h)
        lateMse = lateRmse * lateRmse
        uLate = computeBaseRMSE(lateMse)
        wLate = computeBaseRMSE(lateMse, weights=w)
        uSub = computeBaseRMSE(lateMse[mask])
        wSub = computeBaseRMSE(lateMse[mask], weights=w[mask])
        rows.append(
            {
                "model": name,
                "late_rmse_unweighted": uLate,
                "late_rmse_ipw": wLate,
                "late_rmse_ipw_minus_unweighted": wLate - uLate,
                "policy_subset_late_rmse_unweighted": uSub,
                "policy_subset_late_rmse_ipw": wSub,
                "policy_subset_late_rmse_ipw_minus_unweighted": wSub - uSub,
                "policy_subset_n": int(np.sum(mask)),
            }
        )

    outDf = pd.DataFrame(rows).sort_values("late_rmse_ipw").reset_index(drop=True)
    outCsv = outDir / "table_ipw_vs_unweighted_late_rmse.csv"
    outMd = outDir / "table_ipw_vs_unweighted_late_rmse.md"
    outDf.to_csv(outCsv, index=False)
    saveMdTable(outDf, outMd, precision=4)

    bRows: List[Dict[str, float | str]] = []
    wT = w[mask]
    wC = w[~mask]
    for col in xCols:
        xAll = covDf[col].to_numpy(dtype=np.float64)
        xT = xAll[mask]
        xC = xAll[~mask]
        mtU = float(np.mean(xT))
        mcU = float(np.mean(xC))
        vtU = float(np.var(xT))
        vcU = float(np.var(xC))
        smdU = computeSMD(mtU, vtU, mcU, vcU)
        mtW = weightedMean(xT, wT)
        mcW = weightedMean(xC, wC)
        vtW = weightedVar(xT, wT)
        vcW = weightedVar(xC, wC)
        smdW = computeSMD(mtW, vtW, mcW, vcW)
        bRows.append(
            {
                "covariate": col,
                "treated_mean_unweighted": mtU,
                "control_mean_unweighted": mcU,
                "smd_unweighted": smdU,
                "treated_mean_weighted": mtW,
                "control_mean_weighted": mcW,
                "smd_weighted": smdW,
                "abs_smd_reduction": abs(smdU) - abs(smdW),
            }
        )
    bDf = pd.DataFrame(bRows)
    bCsv = outDir / "balance_check_ipw.csv"
    bMd = outDir / "balance_check_ipw.md"
    bDf.to_csv(bCsv, index=False)
    saveMdTable(bDf, bMd, precision=4)

    wDf = pd.DataFrame(
        {
            "window_index": np.arange(n, dtype=np.int64),
            "is_policy_change_subset": t.astype(int),
            "p_hat_raw": pHatRaw.astype(np.float64),
            "p_hat_clipped": np.asarray(ipw["pHat"], dtype=np.float64),
            "w_raw": np.asarray(ipw["wRaw"], dtype=np.float64),
            "w_clipped": np.asarray(ipw["wClip"], dtype=np.float64),
            "w": w,
            **{cname: covDf[cname].to_numpy(dtype=np.float64) for cname in covDf.columns},
        }
    )
    wCsv = outDir / "ipw_weights_windows.csv"
    wDf.to_csv(wCsv, index=False)

    diag = {
        "eval_dir": str(evalDir),
        "oxford_csv": str(a.oxford_csv),
        "config": str(a.config),
        "n_windows": int(n),
        "horizon": int(h),
        "late_start": int(lateStart),
        "treatment_positive_n": int(np.sum(t)),
        "treatment_positive_ratio": float(np.mean(t)),
        "propensity_model_requested": str(a.propensity_model),
        "propensity_model_fitted": str(fitName),
        "propensity_clip_eps": float(a.propensity_clip_eps),
        "weight_clip_low_q": float(a.weight_clip_low_q),
        "weight_clip_high_q": float(a.weight_clip_high_q),
        "weight_clip_value_low": float(ipw["clipLo"]),
        "weight_clip_value_high": float(ipw["clipHi"]),
        "weight_summary": {
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "q05": float(np.quantile(w, 0.05)),
            "q50": float(np.quantile(w, 0.50)),
            "q95": float(np.quantile(w, 0.95)),
            "effective_sample_size": float((np.sum(w) ** 2) / np.sum(w * w)),
        },
    }
    with open(outDir / "ipw_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print("done")
    print(str(outDir))


if __name__ == "__main__":
    main()

