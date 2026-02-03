import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from crt.config import CRTConfig, NormalizationStats
from crt.model import CRTModel
from crt.data import TimeSeriesCounterfactualDataset
from crt.rollout import rollout
from crt.synthetic import SCMParams, generate_scm_dataset_from_params


def load_model(checkpoint_path: str, config: Optional[CRTConfig] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[CRTModel, CRTConfig, Optional[NormalizationStats], Optional[SCMParams]]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        config: Optional CRTConfig instance. If None, will load from checkpoint.
        device: Device to load model on
        
    Returns:
        Tuple of (loaded CRTModel instance, CRTConfig used, NormalizationStats if available, SCMParams if available)
    """
    # Set weights_only=False to allow loading CRTConfig from checkpoint
    # This is safe since the checkpoint is from our own training
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config from checkpoint if available, otherwise use provided config
    if "config" in checkpoint and checkpoint["config"] is not None:
        config = checkpoint["config"]
        print(f"Loaded config from checkpoint")
    elif config is None:
        raise ValueError("No config found in checkpoint and no config provided")
    else:
        print(f"Using provided config (checkpoint config not found)")
    
    # Load normalization stats from checkpoint if available
    norm_stats = None
    if "norm_stats" in checkpoint and checkpoint["norm_stats"] is not None:
        norm_stats = checkpoint["norm_stats"]
        print(f"Loaded normalization stats from checkpoint")
    else:
        print(f"Warning: No normalization stats found in checkpoint")
    
    # Load SCM params from checkpoint if available
    scm_params = None
    if "scm_params" in checkpoint and checkpoint["scm_params"] is not None:
        scm_state = checkpoint["scm_params"]
        # Reconstruct SCMParams on the current device
        scm_params = SCMParams(
            B_xa=scm_state["B_xa"].to(device),
            W_ax=scm_state["W_ax"].to(device),
            W_aa=scm_state["W_aa"].to(device),
            W_yx=scm_state["W_yx"].to(device),
            W_ya=scm_state["W_ya"].to(device),
            W_yy=scm_state["W_yy"].to(device),
            b_a=scm_state["b_a"].to(device),
        )
        print(f"Loaded SCM parameters from checkpoint")
    else:
        print(f"Warning: No SCM parameters found in checkpoint")
    
    model = CRTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model, config, norm_stats, scm_params


def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    Compute RMSE per horizon step and overall.
    
    Args:
        y_pred: Predicted outcomes of shape (B, H, d_y)
        y_true: True outcomes of shape (B, H, d_y)
        
    Returns:
        Dictionary with:
            - "overall": Overall RMSE across all samples and horizons
            - "per_horizon": List of RMSE values for each horizon step (H values)
    """
    B, H, d_y = y_pred.shape
    
    # Compute squared errors: (B, H, d_y)
    squared_errors = (y_pred - y_true) ** 2
    
    # Overall RMSE: average over all dimensions
    overall_rmse = torch.sqrt(torch.mean(squared_errors)).item()
    
    # Per-horizon RMSE: average over batch and feature dimensions for each horizon step
    per_horizon_rmse = []
    for h in range(H):
        # RMSE for horizon step h: average over batch and features
        horizon_rmse = torch.sqrt(torch.mean(squared_errors[:, h, :])).item()
        per_horizon_rmse.append(horizon_rmse)
    
    return {
        "overall": overall_rmse,
        "per_horizon": per_horizon_rmse
    }


def compute_baseline_rmse(y_true: torch.Tensor, y_mean: torch.Tensor) -> Dict[str, float]:
    """
    Compute baseline RMSE using a naive predictor that always predicts the training-set mean.
    
    This baseline always predicts y_mean (in original scale) for all samples and horizon steps.
    
    Args:
        y_true: True outcomes of shape (N, H, d_y) in original scale
        y_mean: Training-set mean of shape (d_y,) in original scale
        
    Returns:
        Dictionary with:
            - "overall": Overall baseline RMSE
            - "per_horizon": List of baseline RMSE values for each horizon step (H values)
    """
    N, H, d_y = y_true.shape
    
    # Create baseline predictions: always predict y_mean for all samples and horizon steps
    # y_mean is (d_y,), broadcast to (N, H, d_y)
    y_pred_baseline = y_mean.unsqueeze(0).unsqueeze(0).expand(N, H, d_y)  # (N, H, d_y)
    
    # Compute RMSE using the same function
    return compute_rmse(y_pred_baseline, y_true)


def evaluate(
    model: CRTModel,
    test_data: List[Dict[str, torch.Tensor]],
    config: CRTConfig,
    norm_stats: Optional[NormalizationStats] = None,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_results: bool = True,
    save_plots: bool = False,
    plot_dir: str = "evaluation_plots"
):
    """
    Evaluate model on test dataset.
    
    Hides future outcomes and uses rollout() for predictions.
    Computes RMSE per horizon and overall.
    
    Args:
        model: Trained CRTModel instance
        test_data: List of dicts with keys "x", "a", "y" for testing
        config: CRTConfig instance
        norm_stats: Optional normalization statistics. If provided, test data will be normalized
                    and predictions will be denormalized before computing RMSE.
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        plot_results: Whether to plot results
        save_plots: Whether to save plots to disk
        plot_dir: Directory to save plots
    """
    # Create test dataset (with normalization if stats provided)
    test_dataset = TimeSeriesCounterfactualDataset(test_data, config, norm_stats=norm_stats)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Extract batch data
            # DataLoader returns batches with shape (B, 1, T, d_*), so we squeeze the second dimension
            x_hist = batch["x_hist"].squeeze(1).to(device)  # (B, T, d_x)
            a_hist = batch["a_hist"].squeeze(1).to(device)  # (B, T, d_a)
            y_hist = batch["y_hist"].squeeze(1).to(device)  # (B, T, d_y)
            a_fut = batch["a_fut"].squeeze(1).to(device)    # (B, H, d_a)
            y_fut = batch["y_fut"].squeeze(1).to(device)    # (B, H, d_y) - targets (hidden from model)
            
            # Hide future outcomes: call rollout() which doesn't use y_fut
            # This simulates real inference where we don't know future outcomes
            y_pred = rollout(
                model=model,
                x_hist=x_hist,      # (B, T, d_x)
                a_hist=a_hist,     # (B, T, d_a)
                y_hist=y_hist,      # (B, T, d_y)
                a_fut=a_fut         # (B, H, d_a)
            )  # (B, H, d_y) - predictions
            
            all_predictions.append(y_pred.cpu())
            all_targets.append(y_fut.cpu())
    
    # Concatenate all batches: (total_samples, H, d_y)
    y_pred_norm = torch.cat(all_predictions, dim=0)  # (N, H, d_y) - normalized predictions
    y_true_norm = torch.cat(all_targets, dim=0)     # (N, H, d_y) - normalized targets
    
    # Denormalize predictions and targets to original scale for RMSE computation
    if norm_stats is not None:
        # Debug: print normalization stats being used
        print(f"\nEVAL using normalization stats:")
        print(f"  y_mean shape: {norm_stats.y_mean.shape}, range: [{norm_stats.y_mean.min().item():.4f}, {norm_stats.y_mean.max().item():.4f}]")
        print(f"  y_std shape: {norm_stats.y_std.shape}, range: [{norm_stats.y_std.min().item():.4f}, {norm_stats.y_std.max().item():.4f}]")
        
        # Assert y_std is not too small
        assert norm_stats.y_std.min() > 1e-6, f"y_std too small — normalization unstable. Min std: {norm_stats.y_std.min().item():.2e}"
        
        # Denormalize: y = y_norm * y_std + y_mean
        # y_std and y_mean are (d_y,), PyTorch will broadcast to (N, H, d_y) automatically
        y_pred_all = y_pred_norm * norm_stats.y_std + norm_stats.y_mean  # (N, H, d_y)
        y_true_all = y_true_norm * norm_stats.y_std + norm_stats.y_mean  # (N, H, d_y)
        
        # Verify shapes match
        assert y_pred_all.shape == y_pred_norm.shape, f"Shape mismatch after denormalization: {y_pred_all.shape} vs {y_pred_norm.shape}"
        assert y_true_all.shape == y_true_norm.shape, f"Shape mismatch after denormalization: {y_true_all.shape} vs {y_true_norm.shape}"
        
        print(f"Denormalized predictions and targets to original scale for RMSE computation")
        print(f"  y_pred_norm range: [{y_pred_norm.min().item():.4f}, {y_pred_norm.max().item():.4f}]")
        print(f"  y_pred_all range: [{y_pred_all.min().item():.4f}, {y_pred_all.max().item():.4f}]")
    else:
        # No normalization was used
        y_pred_all = y_pred_norm  # (N, H, d_y)
        y_true_all = y_true_norm  # (N, H, d_y)
        print("No normalization stats provided - computing RMSE on raw values")
    
    # Compute RMSE metrics (in original scale)
    rmse_results = compute_rmse(y_pred_all, y_true_all)
    
    # Compute baseline RMSE (naive predictor: always predicts training-set mean)
    if norm_stats is not None:
        # Use training-set mean from normalization stats (in original scale)
        baseline_y_mean = norm_stats.y_mean  # (d_y,)
        print(f"\nComputing baseline RMSE using training-set mean predictor...")
    else:
        # If no normalization stats, compute mean from test data
        # This is less ideal but still provides a baseline
        baseline_y_mean = torch.mean(y_true_all, dim=(0, 1))  # (d_y,) - mean over samples and horizons
        print(f"\nComputing baseline RMSE using test-set mean predictor (no training stats available)...")
    
    baseline_rmse_results = compute_baseline_rmse(y_true_all, baseline_y_mean)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (in original scale)")
    print("="*60)
    
    # Model RMSE
    print(f"\nMODEL RESULTS:")
    print(f"  Overall RMSE: {rmse_results['overall']:.6f}")
    print(f"  RMSE per Horizon Step:")
    for h, rmse_h in enumerate(rmse_results['per_horizon']):
        print(f"    Step {h+1}/{len(rmse_results['per_horizon'])}: {rmse_h:.6f}")
    
    # Baseline RMSE
    print(f"\nBASELINE RESULTS (naive predictor: always predicts mean):")
    print(f"  Overall RMSE (baseline): {baseline_rmse_results['overall']:.6f}")
    print(f"  RMSE per Horizon Step (baseline):")
    for h, rmse_h in enumerate(baseline_rmse_results['per_horizon']):
        print(f"    Step {h+1}/{len(baseline_rmse_results['per_horizon'])}: {rmse_h:.6f}")
    
    # Comparison
    print(f"\nCOMPARISON:")
    improvement_overall = ((baseline_rmse_results['overall'] - rmse_results['overall']) / baseline_rmse_results['overall']) * 100
    if improvement_overall > 0:
        print(f"  Model beats baseline by {improvement_overall:.2f}% (lower is better)")
    else:
        print(f"  Baseline beats model by {abs(improvement_overall):.2f}% (model needs improvement)")
    
    print("="*60)
    
    # Plot results
    if plot_results:
        # Create plot directory if saving
        if save_plots:
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot 1: RMSE per horizon step (with baseline comparison)
        plt.figure(figsize=(10, 6))
        horizons = range(1, len(rmse_results['per_horizon']) + 1)
        plt.plot(horizons, rmse_results['per_horizon'], marker='o', linewidth=2, markersize=8, 
                label='CRT Model', color='blue')
        plt.plot(horizons, baseline_rmse_results['per_horizon'], marker='s', linewidth=2, 
                markersize=8, label='Baseline (mean predictor)', color='red', linestyle='--')
        plt.xlabel('Forecast Horizon Step', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('RMSE per Forecast Horizon Step', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(horizons)
        
        if save_plots:
            plt.savefig(f"{plot_dir}/rmse_per_horizon.png", dpi=300, bbox_inches='tight')
            print(f"\nPlot saved: {plot_dir}/rmse_per_horizon.png")
        else:
            plt.show()
        plt.close()
        
        # Plot 2: Sample predictions vs ground truth (for first few samples)
        # Use denormalized values for plotting (original scale)
        num_samples_to_plot = min(5, y_pred_all.shape[0])
        num_features_to_plot = min(3, y_pred_all.shape[2])  # Plot first 3 features
        
        fig, axes = plt.subplots(num_samples_to_plot, num_features_to_plot, 
                                 figsize=(4*num_features_to_plot, 3*num_samples_to_plot))
        if num_samples_to_plot == 1:
            axes = axes.reshape(1, -1)
        if num_features_to_plot == 1:
            axes = axes.reshape(-1, 1)
        
        for sample_idx in range(num_samples_to_plot):
            for feat_idx in range(num_features_to_plot):
                ax = axes[sample_idx, feat_idx]
                horizons = range(1, y_pred_all.shape[1] + 1)
                
                ax.plot(horizons, y_true_all[sample_idx, :, feat_idx].numpy(), 
                       'b-', label='Ground Truth', linewidth=2, marker='o')
                ax.plot(horizons, y_pred_all[sample_idx, :, feat_idx].numpy(), 
                       'r--', label='Prediction', linewidth=2, marker='s')
                
                ax.set_xlabel('Horizon Step', fontsize=10)
                ax.set_ylabel(f'Feature {feat_idx+1} (original scale)', fontsize=10)
                ax.set_title(f'Sample {sample_idx+1}, Feature {feat_idx+1}', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{plot_dir}/sample_predictions.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_dir}/sample_predictions.png")
        else:
            plt.show()
        plt.close()
    
    return rmse_results


def main():
    """Main evaluation script."""
    # Load model (config, normalization stats, and SCM params will be loaded from checkpoint)
    checkpoint_path = "checkpoints/final_model.pt"  # Update with your checkpoint path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, norm_stats, scm_params = load_model(checkpoint_path, config=None, device=device)
    
    print(f"\nModel config:")
    print(f"  d_x={config.d_x}, d_a={config.d_a}, d_y={config.d_y}")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"  history_len={config.history_len}, forecast_horizon={config.forecast_horizon}")
    
    # Load test data (using synthetic data with the SAME SCM params as training)
    print("\nGenerating synthetic test data using SAME SCM parameters as training...")
    if scm_params is not None:
        # Use the same SCM params from training
        test_data = generate_scm_dataset_from_params(
            params=scm_params,
            n_sequences=100,
            T=64,  # Must be >= history_len + forecast_horizon
            sigma_x=0.05,  # Use same noise levels as training
            sigma_y=0.05,
            device=device
        )
        print(f"Generated {len(test_data)} test sequences using training SCM parameters")
    else:
        # Fallback: generate new params (not ideal, but allows evaluation)
        print("WARNING: No SCM params in checkpoint. Generating test data with new random params.")
        from crt.synthetic import sample_scm_params
        test_scm_params = sample_scm_params(config.d_x, config.d_a, config.d_y, device=device)
        test_data = generate_scm_dataset_from_params(
            params=test_scm_params,
            n_sequences=100,
            T=64,
            sigma_x=0.05,
            sigma_y=0.05,
            device=device
        )
        print(f"Generated {len(test_data)} test sequences with new SCM parameters")
    
    # Evaluate
    rmse_results = evaluate(
        model=model,
        test_data=test_data,
        config=config,
        norm_stats=norm_stats,  # Pass normalization stats for denormalization
        batch_size=32,
        device=device,
        plot_results=True,
        save_plots=True,
        plot_dir="evaluation_plots"
    )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

