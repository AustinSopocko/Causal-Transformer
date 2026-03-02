from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from crt.rollout import rollout
from src.data.normalise import OutcomeScaler, inverse_transform_outcomes


def make_alternative_policy_path(
    observed_a_fut: torch.Tensor,
    mode: str = "higher_stringency",
    scale: float = 1.2,
    delay_steps: int = 3,
) -> torch.Tensor:
    """Build a simple alternative future policy path for qualitative checks."""
    alt = observed_a_fut.clone()

    if mode == "higher_stringency":
        alt = alt * scale
    elif mode == "delayed_relaxation":
        if delay_steps > 0 and alt.shape[0] > delay_steps:
            alt[delay_steps:] = alt[:-delay_steps]
    else:
        raise ValueError(f"Unknown alternative mode: {mode}")

    return alt


def rollout_policy_paths(
    model,
    x_hist: torch.Tensor,
    a_hist: torch.Tensor,
    y_hist: torch.Tensor,
    observed_a_fut: torch.Tensor,
    alternative_a_fut: torch.Tensor,
    country_idx: Optional[torch.Tensor] = None,
    scaler: Optional[OutcomeScaler] = None,
    device: str = "cpu",
    use_future_policy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out two policy futures from the same historical context."""
    model.eval()

    x_hist = x_hist.to(device)
    a_hist = a_hist.to(device)
    y_hist = y_hist.to(device)
    observed_a_fut = observed_a_fut.to(device)
    alternative_a_fut = alternative_a_fut.to(device)
    country_idx = country_idx.to(device) if country_idx is not None else None

    with torch.no_grad():
        y_obs = rollout(
            model=model,
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=observed_a_fut,
            country_idx=country_idx,
            use_future_policy=use_future_policy,
        )
        y_alt = rollout(
            model=model,
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=alternative_a_fut,
            country_idx=country_idx,
            use_future_policy=use_future_policy,
        )

    if scaler is not None:
        y_obs = inverse_transform_outcomes(y_obs.cpu(), scaler)
        y_alt = inverse_transform_outcomes(y_alt.cpu(), scaler)
    else:
        y_obs = y_obs.cpu()
        y_alt = y_alt.cpu()

    return y_obs, y_alt


def plot_counterfactual_trajectories(
    observed_pred: torch.Tensor,
    alternative_pred: torch.Tensor,
    outcome_names: Optional[Iterable[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Counterfactual policy rollout",
) -> None:
    """Plot side-by-side predicted trajectories under two policy futures."""
    if observed_pred.ndim != 2 or alternative_pred.ndim != 2:
        raise ValueError("Expected tensors with shape (H, d_y)")

    if observed_pred.shape != alternative_pred.shape:
        raise ValueError("Observed and alternative predictions must have matching shapes")

    horizon, d_y = observed_pred.shape
    names: List[str] = list(outcome_names) if outcome_names is not None else [f"outcome_{i}" for i in range(d_y)]

    fig, axes = plt.subplots(d_y, 1, figsize=(10, 3 * d_y), sharex=True)
    if d_y == 1:
        axes = [axes]

    x = list(range(1, horizon + 1))
    for idx, ax in enumerate(axes):
        ax.plot(x, observed_pred[:, idx], label="Observed policy path", linewidth=2.0)
        ax.plot(x, alternative_pred[:, idx], label="Alternative policy path", linewidth=2.0, linestyle="--")
        ax.set_ylabel(names[idx])
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.set_title(title)

    axes[-1].set_xlabel("Forecast step")
    axes[0].legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)

    plt.close(fig)
