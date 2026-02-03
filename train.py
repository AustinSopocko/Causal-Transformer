import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import os
from pathlib import Path

from crt.config import CRTConfig, NormalizationStats
from crt.model import CRTModel
from crt.data import TimeSeriesCounterfactualDataset, make_synthetic_dataset, compute_normalization_stats
from crt.synthetic import SCMParams


def load_dataset_and_dataloader(
    cfg: CRTConfig,
    n_sequences: int = 500,
    T: int = 64,
    batch_size: int = 32,
    shuffle: bool = True,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
) -> Tuple[DataLoader, NormalizationStats, "SCMParams"]:
    """
    Creates a synthetic SCM dataset and wraps it in a DataLoader.
    
    Uses the synthetic Structural Causal Model (SCM) generator to create
    synthetic time series data for training.
    
    Args:
        cfg: CRTConfig instance containing model and dataset hyperparameters
        n_sequences: Number of independent trajectories to generate (default: 500)
        T: Total length of each trajectory (default: 64)
            Must satisfy: history_len + forecast_horizon <= T
        batch_size: DataLoader batch size (default: 32)
        shuffle: Whether to shuffle the dataset (default: True)
        sigma_x: Standard deviation of noise for state transitions (default: 0.1)
        sigma_y: Standard deviation of noise for outcome transitions (default: 0.1)
        
    Returns:
        Tuple of (DataLoader, NormalizationStats):
            - DataLoader that yields batches with keys:
                - x_hist: (B, history_len, d_x) - historical states
                - a_hist: (B, history_len, d_a) - historical actions
                - y_hist: (B, history_len, d_y) - historical outcomes (normalized)
                - a_fut: (B, forecast_horizon, d_a) - future actions
                - y_fut: (B, forecast_horizon, d_y) - future outcomes (normalized, targets)
            - NormalizationStats: Statistics for denormalizing predictions
        
    Raises:
        AssertionError: If history_len + forecast_horizon > T
    """
    # Assert that sequence length is sufficient
    assert cfg.history_len + cfg.forecast_horizon <= T, (
        f"Sequence length T={T} must be at least history_len + forecast_horizon = "
        f"{cfg.history_len} + {cfg.forecast_horizon} = {cfg.history_len + cfg.forecast_horizon}"
    )
    
    # Generate synthetic SCM dataset using make_synthetic_dataset
    # This will sample SCM params if not provided, and return both dataset and params
    from crt.data import make_synthetic_dataset
    print(f"Generating {n_sequences} sequences of length T={T} with sigma_x={sigma_x}, sigma_y={sigma_y}...")
    
    # First generate data without normalization to compute stats
    # We'll create a temporary dataset to get the raw data
    from crt.synthetic import sample_scm_params, generate_scm_dataset_from_params
    scm_params = sample_scm_params(cfg.d_x, cfg.d_a, cfg.d_y)
    synthetic_data = generate_scm_dataset_from_params(
        params=scm_params,
        n_sequences=n_sequences,
        T=T,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )
    
    # Print dataset summary
    print(f"\nSynthetic dataset summary: {n_sequences} sequences, T={T}")
    
    # Compute normalization statistics BEFORE windowing (on full sequences)
    # This ensures stats are computed on the complete training set
    print("\nComputing normalization statistics from full training sequences...")
    norm_stats = compute_normalization_stats(synthetic_data)
    print(f"Normalization stats computed: y_mean shape {norm_stats.y_mean.shape}, y_std shape {norm_stats.y_std.shape}")
    
    # Create dataset with normalization (normalization applied during windowing)
    # Use the same scm_params we used to generate the data
    dataset, _ = make_synthetic_dataset(
        cfg=cfg,
        n_sequences=n_sequences,
        T=T,
        norm_stats=norm_stats,
        params=scm_params,  # Use the same params
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )
    
    # Wrap in DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False  # Set to True if using CUDA
    )
    
    return dataloader, norm_stats, scm_params


def train(
    config: CRTConfig,
    train_loader: DataLoader,
    norm_stats: NormalizationStats,
    scm_params: SCMParams,
    num_epochs: int = 100,
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the CRT model.
    
    Args:
        config: CRTConfig instance with model hyperparameters
        train_loader: DataLoader for training data (y values are already normalized)
        norm_stats: NormalizationStats for denormalizing predictions if needed
        scm_params: SCMParams used to generate training data (will be saved in checkpoint)
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        device: Device to train on ("cuda" or "cpu")
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # 3. Initialize CRTConfig and CRTModel
    print("Initializing model...")
    model = CRTModel(config).to(device)
    print(f"Model initialized on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Use Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr
    )
    
    # 5. MSE loss for regression
    criterion = nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 6. Implement linear teacher forcing schedule
        # tf_ratio = start + (end - start) * (epoch / num_epochs)
        tf_ratio = config.teacher_forcing_start + (
            config.teacher_forcing_end - config.teacher_forcing_start
        ) * (epoch / num_epochs)
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract batch data
            # DataLoader returns batches with shape (B, 1, T, d_*), so we squeeze the second dimension
            x_hist = batch["x_hist"].squeeze(1).to(device)  # (B, T, d_x)
            a_hist = batch["a_hist"].squeeze(1).to(device)  # (B, T, d_a)
            y_hist = batch["y_hist"].squeeze(1).to(device)  # (B, T, d_y)
            a_fut = batch["a_fut"].squeeze(1).to(device)    # (B, H, d_a)
            y_fut = batch["y_fut"].squeeze(1).to(device)    # (B, H, d_y) - targets
            
            # Forward pass
            # Training mode: provide y_fut for teacher forcing/scheduled sampling
            y_pred = model(
                x_hist=x_hist,      # (B, T, d_x)
                a_hist=a_hist,       # (B, T, d_a)
                y_hist=y_hist,       # (B, T, d_y)
                a_fut=a_fut,         # (B, H, d_a)
                y_fut=y_fut,         # (B, H, d_y) - for teacher forcing
                teacher_forcing_prob=tf_ratio
            )  # (B, H, d_y)
            
            # Compute loss
            loss = criterion(y_pred, y_fut)  # MSE loss: (B, H, d_y) vs (B, H, d_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss: {avg_loss:.6f} | "
            f"Teacher Forcing: {tf_ratio:.3f}"
        )
        
        # 7. Save model checkpoints
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            # Prepare SCM params dict for saving (move to CPU)
            scm_params_dict = {
                "B_xa": scm_params.B_xa.cpu(),
                "W_ax": scm_params.W_ax.cpu(),
                "W_aa": scm_params.W_aa.cpu(),
                "W_yx": scm_params.W_yx.cpu(),
                "W_ya": scm_params.W_ya.cpu(),
                "W_yy": scm_params.W_yy.cpu(),
                "b_a": scm_params.b_a.cpu(),
            }
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
                "norm_stats": norm_stats,  # Save normalization stats
                "scm_params": scm_params_dict,  # Save SCM parameters
                "teacher_forcing_ratio": tf_ratio
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("\nTraining completed!")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    
    # Prepare SCM params dict for saving (move to CPU)
    scm_params_dict = {
        "B_xa": scm_params.B_xa.cpu(),
        "W_ax": scm_params.W_ax.cpu(),
        "W_aa": scm_params.W_aa.cpu(),
        "W_yx": scm_params.W_yx.cpu(),
        "W_ya": scm_params.W_ya.cpu(),
        "W_yy": scm_params.W_yy.cpu(),
        "b_a": scm_params.b_a.cpu(),
    }
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "norm_stats": norm_stats,  # Save normalization stats
        "scm_params": scm_params_dict,  # Save SCM parameters
        "num_epochs": num_epochs
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")


def main():
    """Main training script."""
    # Initialize config with improved model capacity for better performance
    config = CRTConfig(
        d_x=8,           # State dimension
        d_a=4,           # Action dimension
        d_y=8,           # Outcome dimension
        d_model=64,      # Increased model dimension for better expressiveness
        n_heads=4,       # Increased number of heads
        n_layers_enc=3,  # Increased number of encoder layers
        n_layers_dec=3,  # Increased number of decoder layers
        history_len=10,  # History window length
        forecast_horizon=5,  # Forecast horizon
        dropout=0.1,
        lr=1e-4,        # Learning rate (slightly smaller for stability with bigger model)
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.0
    )
    
    # Print model config summary
    print("="*60)
    print("CRT MODEL CONFIGURATION")
    print("="*60)
    print(f"  d_x={config.d_x}, d_a={config.d_a}, d_y={config.d_y}")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"  n_layers_enc={config.n_layers_enc}, n_layers_dec={config.n_layers_dec}")
    print(f"  history_len={config.history_len}, forecast_horizon={config.forecast_horizon}")
    print(f"  dropout={config.dropout}, lr={config.lr}")
    print("="*60)
    
    # Create synthetic dataset and DataLoader
    print("\nCreating synthetic SCM dataset...")
    train_loader, norm_stats, scm_params = load_dataset_and_dataloader(
        cfg=config,
        n_sequences=400,  # Moderate dataset size
        T=64,            # Sequence length (must be >= history_len + forecast_horizon = 15)
        batch_size=64,   # Increased batch size for better gradient estimates
        shuffle=True,
        sigma_x=0.05,    # Reduced noise for cleaner signal
        sigma_y=0.05     # Reduced noise for cleaner signal
    )
    
    # Train model
    train(
        config=config,
        train_loader=train_loader,
        norm_stats=norm_stats,
        scm_params=scm_params,  # Pass SCM params to save in checkpoint
        num_epochs=40,  # 4x the previous number of epochs
        checkpoint_dir="checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    main()

