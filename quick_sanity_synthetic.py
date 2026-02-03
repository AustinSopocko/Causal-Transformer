"""
Quick sanity check script for synthetic SCM data generation and CRTModel.

Verifies that:
1. Synthetic data generation works
2. Dataset and DataLoader are wired correctly
3. CRTModel can process the data and produce correct output shapes
"""

import torch
from crt.config import CRTConfig
from crt.model import CRTModel
from train import load_dataset_and_dataloader


def main():
    """Run sanity checks on synthetic data and CRTModel."""
    print("=" * 60)
    print("Quick Sanity Check: Synthetic SCM Data + CRTModel")
    print("=" * 60)
    
    # 1. Create small CRTConfig
    print("\n1. Creating small CRTConfig...")
    cfg = CRTConfig(
        d_x=4,              # Small state dimension
        d_a=2,              # Small action dimension
        d_y=3,              # Small outcome dimension
        d_model=64,         # Small model dimension
        n_heads=4,          # Number of attention heads
        n_layers_enc=2,     # Small number of encoder layers
        n_layers_dec=2,     # Small number of decoder layers
        history_len=16,     # History window length
        forecast_horizon=8, # Forecast horizon
        dropout=0.1,
        lr=1e-4,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.0
    )
    print(f"   Config created:")
    print(f"   - d_x={cfg.d_x}, d_a={cfg.d_a}, d_y={cfg.d_y}")
    print(f"   - d_model={cfg.d_model}, n_heads={cfg.n_heads}")
    print(f"   - history_len={cfg.history_len}, forecast_horizon={cfg.forecast_horizon}")
    
    # 2. Create DataLoader with synthetic data
    print("\n2. Creating synthetic dataset and DataLoader...")
    T = 32  # Sequence length (must be >= history_len + forecast_horizon = 24)
    train_loader = load_dataset_and_dataloader(
        cfg=cfg,
        n_sequences=100,  # Small number for quick test
        T=T,
        batch_size=8,     # Small batch size
        shuffle=True
    )
    print(f"   Dataset size: {len(train_loader.dataset)} samples")
    print(f"   Batches: {len(train_loader)}")
    
    # 3. Fetch one batch and print shapes
    print("\n3. Fetching one batch and checking shapes...")
    batch = next(iter(train_loader))
    
    # Extract batch data (squeeze the extra dimension from dataset)
    x_hist = batch["x_hist"].squeeze(1)  # (B, history_len, d_x)
    a_hist = batch["a_hist"].squeeze(1)  # (B, history_len, d_a)
    y_hist = batch["y_hist"].squeeze(1)  # (B, history_len, d_y)
    a_fut = batch["a_fut"].squeeze(1)    # (B, forecast_horizon, d_a)
    y_fut = batch["y_fut"].squeeze(1)    # (B, forecast_horizon, d_y)
    
    B = x_hist.shape[0]
    print(f"   Batch size: {B}")
    print(f"   x_hist shape: {x_hist.shape} (expected: ({B}, {cfg.history_len}, {cfg.d_x}))")
    print(f"   a_hist shape: {a_hist.shape} (expected: ({B}, {cfg.history_len}, {cfg.d_a}))")
    print(f"   y_hist shape: {y_hist.shape} (expected: ({B}, {cfg.history_len}, {cfg.d_y}))")
    print(f"   a_fut shape: {a_fut.shape} (expected: ({B}, {cfg.forecast_horizon}, {cfg.d_a}))")
    print(f"   y_fut shape: {y_fut.shape} (expected: ({B}, {cfg.forecast_horizon}, {cfg.d_y}))")
    
    # Verify shapes
    assert x_hist.shape == (B, cfg.history_len, cfg.d_x), f"x_hist shape mismatch: {x_hist.shape}"
    assert a_hist.shape == (B, cfg.history_len, cfg.d_a), f"a_hist shape mismatch: {a_hist.shape}"
    assert y_hist.shape == (B, cfg.history_len, cfg.d_y), f"y_hist shape mismatch: {y_hist.shape}"
    assert a_fut.shape == (B, cfg.forecast_horizon, cfg.d_a), f"a_fut shape mismatch: {a_fut.shape}"
    assert y_fut.shape == (B, cfg.forecast_horizon, cfg.d_y), f"y_fut shape mismatch: {y_fut.shape}"
    print("   ✓ All batch shapes are correct!")
    
    # 4. Instantiate CRTModel and run forward pass
    print("\n4. Instantiating CRTModel...")
    model = CRTModel(cfg)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n5. Running forward pass...")
    model.eval()
    with torch.no_grad():
        y_pred = model(
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=a_fut,
            y_fut=y_fut  # Training mode with teacher forcing
        )
    
    print(f"   y_pred shape: {y_pred.shape}")
    print(f"   Expected shape: ({B}, {cfg.forecast_horizon}, {cfg.d_y})")
    
    # Verify output shape
    expected_shape = (B, cfg.forecast_horizon, cfg.d_y)
    assert y_pred.shape == expected_shape, \
        f"Output shape mismatch! Got {y_pred.shape}, expected {expected_shape}"
    print("   ✓ Output shape is correct!")
    
    # 6. Test inference mode (without y_fut)
    print("\n6. Testing inference mode (autoregressive rollout)...")
    with torch.no_grad():
        y_pred_inf = model(
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=a_fut,
            y_fut=None  # Inference mode
        )
    
    print(f"   y_pred_inf shape: {y_pred_inf.shape}")
    assert y_pred_inf.shape == expected_shape, \
        f"Inference output shape mismatch! Got {y_pred_inf.shape}, expected {expected_shape}"
    print("   ✓ Inference mode output shape is correct!")
    
    print("\n" + "=" * 60)
    print("All sanity checks passed! ✓")
    print("=" * 60)
    print("\nThe synthetic data generator, dataset, and CRTModel are wired correctly.")


if __name__ == "__main__":
    main()

