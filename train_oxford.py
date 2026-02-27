import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from crt.baselines import GRUBaseline, LinearBaseline, MLPBaseline, TCNBaseline
from crt.config import CRTConfig
from crt.model import CRTModel
from crt.rollout import rollout
from src.data.context_builder import build_country_index
from src.data.normalise import OutcomeScaler, apply_outcome_scaler, fit_outcome_scaler, inverse_transform_outcomes
from src.data.oxford_loader import clean_oxford, load_oxford_csv, panel_stats, select_features
from src.data.panel_windows import OxfordPanelDataset, describe_window_shapes, make_windows
from src.train.splits import heldout_country_window_indices, time_split_window_indices


def load_yaml_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_config(
    cfg: Dict,
    d_x: int,
    d_a: int,
    d_y: int,
    num_countries: int,
    use_country_context: bool,
    use_future_policy: bool,
    rollout_training: bool,
) -> CRTConfig:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    window_cfg = cfg["window"]

    return CRTConfig(
        d_x=d_x,
        d_a=d_a,
        d_y=d_y,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers_enc=int(model_cfg["n_layers_enc"]),
        n_layers_dec=int(model_cfg["n_layers_dec"]),
        history_len=int(window_cfg["history_len"]),
        forecast_horizon=int(window_cfg["forecast_horizon"]),
        dropout=float(model_cfg["dropout"]),
        lr=float(train_cfg["lr"]),
        teacher_forcing_start=float(train_cfg["teacher_forcing_start"]),
        teacher_forcing_end=float(train_cfg["teacher_forcing_end"]),
        num_countries=int(max(1, num_countries)),
        use_country_context=use_country_context,
        use_future_policy=use_future_policy,
        rollout_training=rollout_training,
    )


def evaluate_rmse(
    model,
    loader: DataLoader,
    scaler: OutcomeScaler,
    model_type: str,
    device: str,
    use_future_policy: bool,
) -> Dict[str, object]:
    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for batch in loader:
            x_hist = batch["x_hist"].to(device)
            a_hist = batch["a_hist"].to(device)
            y_hist = batch["y_hist"].to(device)
            a_fut = batch["a_fut"].to(device)
            y_fut = batch["y_fut"].to(device)
            country_idx = batch["country_idx"].to(device)

            if model_type == "crt":
                y_pred = rollout(
                    model=model,
                    x_hist=x_hist,
                    a_hist=a_hist,
                    y_hist=y_hist,
                    a_fut=a_fut,
                    country_idx=country_idx,
                    use_future_policy=use_future_policy,
                )
            else:
                y_pred = model(
                    x_hist=x_hist,
                    a_hist=a_hist,
                    y_hist=y_hist,
                    a_fut=a_fut,
                    y_fut=None,
                )

            y_pred_all.append(inverse_transform_outcomes(y_pred.cpu(), scaler))
            y_true_all.append(inverse_transform_outcomes(y_fut.cpu(), scaler))

    y_pred_full = torch.cat(y_pred_all, dim=0)
    y_true_full = torch.cat(y_true_all, dim=0)
    sq_err = (y_pred_full - y_true_full) ** 2

    overall = torch.sqrt(torch.mean(sq_err)).item()
    per_horizon = [torch.sqrt(torch.mean(sq_err[:, h, :])).item() for h in range(sq_err.shape[1])]
    return {"overall": overall, "per_horizon": per_horizon}


def build_model(model_type: str, config: CRTConfig):
    if model_type == "crt":
        return CRTModel(config)
    if model_type == "linear":
        return LinearBaseline(config)
    if model_type == "mlp":
        return MLPBaseline(config, hidden_size=64)
    if model_type == "gru":
        return GRUBaseline(config, hidden_size=64, num_layers=1)
    if model_type == "tcn":
        return TCNBaseline(config, num_filters=64, kernel_size=3)
    raise ValueError(f"Unsupported model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRT on Oxford panel windows")
    parser.add_argument("--oxford_csv", type=str, required=True, help="Path to Oxford panel CSV")
    parser.add_argument("--config", type=str, default="src/configs/oxford_config.yaml", help="Oxford config YAML")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/oxford", help="Output checkpoint directory")
    parser.add_argument("--model_type", type=str, default="crt", choices=["crt", "linear", "mlp", "gru", "tcn"])
    parser.add_argument("--split_strategy", type=str, default=None, choices=["time", "country"])
    parser.add_argument("--train_fraction", type=float, default=None)
    parser.add_argument("--holdout_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drop_nan_windows", action="store_true", help="Drop windows with NaNs")
    parser.add_argument("--keep_nan_windows", action="store_true", help="Keep windows even if NaNs exist")
    parser.add_argument("--log1p_outcomes", action="store_true", help="Apply log1p before outcome normalization")
    parser.add_argument("--no_country_context", action="store_true", help="Disable country embedding context")
    parser.add_argument("--no_future_policy", action="store_true", help="Disable policy conditioning")
    parser.add_argument("--no_rollout", action="store_true", help="Disable autoregressive rollout during training")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    dataset_cfg = cfg["dataset"]
    window_cfg = cfg["window"]
    split_cfg = cfg["split"]
    train_cfg = cfg["training"]
    norm_cfg = cfg["normalization"]

    split_strategy = args.split_strategy or split_cfg.get("strategy", "time")
    train_fraction = args.train_fraction if args.train_fraction is not None else float(split_cfg.get("train_fraction", 0.8))
    holdout_fraction = args.holdout_fraction if args.holdout_fraction is not None else float(split_cfg.get("holdout_fraction", 0.2))
    seed = args.seed if args.seed is not None else int(split_cfg.get("seed", 42))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 64))
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 40))

    history_len = int(window_cfg["history_len"])
    forecast_horizon = int(window_cfg["forecast_horizon"])
    stride = int(window_cfg["stride"])

    policy_cols = list(dataset_cfg["policy_cols"])
    outcome_cols = list(dataset_cfg["outcome_cols"])
    state_cols = list(dataset_cfg.get("state_cols", []))

    raw_df = load_oxford_csv(args.oxford_csv)
    cleaned = clean_oxford(
        raw_df,
        country_col=dataset_cfg.get("country_col", "CountryName"),
        date_col=dataset_cfg.get("date_col", "Date"),
    )
    panel_df = select_features(
        cleaned,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols,
    )

    if len(state_cols) == 0:
        panel_df["__dummy_state__"] = 0.0
        state_cols = ["__dummy_state__"]

    stats, missingness = panel_stats(panel_df, policy_cols + outcome_cols + state_cols)
    print("=" * 60)
    print("OXFORD PANEL SUMMARY")
    print("=" * 60)
    print(f"Rows: {stats['n_rows']}")
    print(f"Countries: {stats['n_countries']}")
    print(f"Date range: {stats['date_min']} -> {stats['date_max']}")
    if not missingness.empty:
        print("Missingness (top 10):")
        print(missingness.head(10).to_string())

    country_to_idx = build_country_index(panel_df)

    drop_nan_windows = True
    if args.keep_nan_windows:
        drop_nan_windows = False
    elif args.drop_nan_windows:
        drop_nan_windows = True

    windows = make_windows(
        panel_df,
        history_len=history_len,
        forecast_horizon=forecast_horizon,
        stride=stride,
        policy_cols=policy_cols,
        outcome_cols=outcome_cols,
        state_cols=state_cols,
        country_to_idx=country_to_idx,
        drop_nan_windows=drop_nan_windows,
    )

    print("Window shapes:")
    for key, shape in describe_window_shapes(windows).items():
        print(f"  {key}: {shape}")

    if len(windows) == 0:
        raise RuntimeError("No windows were produced. Check feature coverage, T/H/stride, and missingness.")

    if split_strategy == "time":
        train_idx, test_idx, cutoff_dates = time_split_window_indices(windows.metadata, train_fraction=train_fraction)
        print(f"Time split cutoffs generated for {len(cutoff_dates)} countries")
    elif split_strategy == "country":
        train_idx, test_idx, heldout = heldout_country_window_indices(
            windows.metadata,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        print(f"Held-out countries ({len(heldout)}): {', '.join(heldout[:10])}")
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            f"Split produced empty set(s): train={train_idx.size}, test={test_idx.size}. "
            "Adjust split ratio, window sizes, or dataset coverage."
        )

    train_windows = windows.subset(train_idx)
    test_windows = windows.subset(test_idx)

    log1p_outcomes = args.log1p_outcomes or bool(norm_cfg.get("log1p_outcomes", False))
    scaler = fit_outcome_scaler(train_windows, log1p=log1p_outcomes)

    train_windows = apply_outcome_scaler(train_windows, scaler)
    test_windows = apply_outcome_scaler(test_windows, scaler)

    train_loader = DataLoader(OxfordPanelDataset(train_windows), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(OxfordPanelDataset(test_windows), batch_size=batch_size, shuffle=False, num_workers=0)

    model_cfg = build_model_config(
        cfg=cfg,
        d_x=len(state_cols),
        d_a=len(policy_cols),
        d_y=len(outcome_cols),
        num_countries=len(country_to_idx),
        use_country_context=not args.no_country_context,
        use_future_policy=not args.no_future_policy,
        rollout_training=not args.no_rollout,
    )

    print("=" * 60)
    print("MODEL CONFIG")
    print("=" * 60)
    print(
        f"d_x={model_cfg.d_x}, d_a={model_cfg.d_a}, d_y={model_cfg.d_y}, "
        f"num_countries={model_cfg.num_countries}"
    )
    print(
        f"country_context={model_cfg.use_country_context}, future_policy={model_cfg.use_future_policy}, "
        f"rollout_training={model_cfg.rollout_training}"
    )

    model = build_model(args.model_type, model_cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr)
    criterion = nn.MSELoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        tf_ratio = model_cfg.teacher_forcing_start + (
            model_cfg.teacher_forcing_end - model_cfg.teacher_forcing_start
        ) * (epoch / max(epochs, 1))

        for batch in train_loader:
            x_hist = batch["x_hist"].to(args.device)
            a_hist = batch["a_hist"].to(args.device)
            y_hist = batch["y_hist"].to(args.device)
            a_fut = batch["a_fut"].to(args.device)
            y_fut = batch["y_fut"].to(args.device)
            country_idx = batch["country_idx"].to(args.device)

            if args.model_type == "crt":
                y_pred = model(
                    x_hist=x_hist,
                    a_hist=a_hist,
                    y_hist=y_hist,
                    a_fut=a_fut,
                    y_fut=y_fut,
                    teacher_forcing_prob=tf_ratio,
                    country_idx=country_idx,
                    use_future_policy=not args.no_future_policy,
                    rollout_training=not args.no_rollout,
                )
            else:
                y_pred = model(
                    x_hist=x_hist,
                    a_hist=a_hist,
                    y_hist=y_hist,
                    a_fut=a_fut,
                    y_fut=y_fut,
                    teacher_forcing_prob=tf_ratio,
                )

            loss = criterion(y_pred, y_fut)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)

        rmse = evaluate_rmse(
            model=model,
            loader=test_loader,
            scaler=scaler,
            model_type=args.model_type,
            device=args.device,
            use_future_policy=not args.no_future_policy,
        )

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"test_rmse={rmse['overall']:.6f} | tf={tf_ratio:.3f}"
        )

        if rmse["overall"] < best_rmse:
            best_rmse = rmse["overall"]
            best_path = checkpoint_dir / f"best_{args.model_type}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                    "scaler": scaler,
                    "country_to_idx": country_to_idx,
                    "policy_cols": policy_cols,
                    "outcome_cols": outcome_cols,
                    "state_cols": state_cols,
                    "split_strategy": split_strategy,
                    "rmse": rmse,
                    "model_type": args.model_type,
                },
                best_path,
            )

    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_rmse = evaluate_rmse(
        model=model,
        loader=test_loader,
        scaler=scaler,
        model_type=args.model_type,
        device=args.device,
        use_future_policy=not args.no_future_policy,
    )
    print(f"Overall RMSE: {final_rmse['overall']:.6f}")
    for idx, value in enumerate(final_rmse["per_horizon"], start=1):
        print(f"  Step {idx}: {value:.6f}")


if __name__ == "__main__":
    main()
