"""
Test script to verify CRTModel instantiation and forward pass.

Creates a model with small dimensions, runs forward pass with random tensors,
and prints output shapes.
"""

import torch
from crt.config import CRTConfig
from crt.model import CRTModel


def test_crt_model():
    """Test CRTModel with small dimensions."""
    print("=" * 60)
    print("Testing CRTModel")
    print("=" * 60)
    
    # Create config with small dimensions for testing
    config = CRTConfig(
        d_x=8,           # Small state dimension
        d_a=4,           # Small action dimension
        d_y=8,           # Small outcome dimension
        d_model=16,      # Small model dimension
        n_heads=2,       # Small number of heads
        n_layers_enc=2,  # Small number of encoder layers
        n_layers_dec=2,  # Small number of decoder layers
        history_len=5,   # Short history
        forecast_horizon=3,  # Short forecast horizon
        dropout=0.1,
        lr=1e-4,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.0
    )
    
    print(f"\nConfig:")
    print(f"  d_x={config.d_x}, d_a={config.d_a}, d_y={config.d_y}")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}")
    print(f"  history_len={config.history_len}, forecast_horizon={config.forecast_horizon}")
    
    # Instantiate model
    print("\nInstantiating CRTModel...")
    model = CRTModel(config)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create random input tensors
    batch_size = 2
    device = "cpu"
    
    print(f"\nCreating random input tensors (batch_size={batch_size})...")
    x_hist = torch.randn(batch_size, config.history_len, config.d_x, device=device)
    a_hist = torch.randn(batch_size, config.history_len, config.d_a, device=device)
    y_hist = torch.randn(batch_size, config.history_len, config.d_y, device=device)
    a_fut = torch.randn(batch_size, config.forecast_horizon, config.d_a, device=device)
    y_fut = torch.randn(batch_size, config.forecast_horizon, config.d_y, device=device)
    
    print(f"  x_hist: {x_hist.shape}")
    print(f"  a_hist: {a_hist.shape}")
    print(f"  y_hist: {y_hist.shape}")
    print(f"  a_fut: {a_fut.shape}")
    print(f"  y_fut: {y_fut.shape}")
    
    # Test 1: Training mode (with y_fut)
    print("\n" + "-" * 60)
    print("Test 1: Training mode (with teacher forcing)")
    print("-" * 60)
    model.train()
    y_pred_train = model(
        x_hist=x_hist,
        a_hist=a_hist,
        y_hist=y_hist,
        a_fut=a_fut,
        y_fut=y_fut,
        teacher_forcing_prob=1.0
    )
    print(f"Output shape: {y_pred_train.shape}")
    print(f"Expected shape: ({batch_size}, {config.forecast_horizon}, {config.d_y})")
    assert y_pred_train.shape == (batch_size, config.forecast_horizon, config.d_y), \
        f"Shape mismatch! Got {y_pred_train.shape}, expected ({batch_size}, {config.forecast_horizon}, {config.d_y})"
    print("✓ Training mode test passed!")
    
    # Test 2: Inference mode (without y_fut)
    print("\n" + "-" * 60)
    print("Test 2: Inference mode (autoregressive rollout)")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        y_pred_inf = model(
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=a_fut,
            y_fut=None  # Inference mode
        )
    print(f"Output shape: {y_pred_inf.shape}")
    print(f"Expected shape: ({batch_size}, {config.forecast_horizon}, {config.d_y})")
    assert y_pred_inf.shape == (batch_size, config.forecast_horizon, config.d_y), \
        f"Shape mismatch! Got {y_pred_inf.shape}, expected ({batch_size}, {config.forecast_horizon}, {config.d_y})"
    print("✓ Inference mode test passed!")
    
    # Test 3: With attention visualization
    print("\n" + "-" * 60)
    print("Test 3: With attention visualization")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        y_pred_attn, attention_dict = model(
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=a_fut,
            y_fut=None,
            return_attention=True
        )
    print(f"Output shape: {y_pred_attn.shape}")
    print(f"Expected shape: ({batch_size}, {config.forecast_horizon}, {config.d_y})")
    assert y_pred_attn.shape == (batch_size, config.forecast_horizon, config.d_y), \
        f"Shape mismatch! Got {y_pred_attn.shape}, expected ({batch_size}, {config.forecast_horizon}, {config.d_y})"
    
    print(f"\nAttention shapes:")
    if attention_dict["encoder_attn"]:
        print(f"  encoder_attn: {len(attention_dict['encoder_attn'])} layers")
        for i, attn in enumerate(attention_dict["encoder_attn"]):
            print(f"    Layer {i}: {attn.shape}")
    
    if attention_dict["decoder_self_attn"]:
        print(f"  decoder_self_attn: {len(attention_dict['decoder_self_attn'])} layers")
        for i, attn in enumerate(attention_dict["decoder_self_attn"]):
            print(f"    Layer {i}: {attn.shape}")
    
    if attention_dict["decoder_cross_attn"]:
        print(f"  decoder_cross_attn: {len(attention_dict['decoder_cross_attn'])} layers")
        for i, attn in enumerate(attention_dict["decoder_cross_attn"]):
            print(f"    Layer {i}: {attn.shape}")
    
    print("✓ Attention visualization test passed!")
    
    # Test 4: Scheduled sampling
    print("\n" + "-" * 60)
    print("Test 4: Scheduled sampling (teacher_forcing_prob=0.5)")
    print("-" * 60)
    model.train()
    y_pred_sched = model(
        x_hist=x_hist,
        a_hist=a_hist,
        y_hist=y_hist,
        a_fut=a_fut,
        y_fut=y_fut,
        teacher_forcing_prob=0.5
    )
    print(f"Output shape: {y_pred_sched.shape}")
    print(f"Expected shape: ({batch_size}, {config.forecast_horizon}, {config.d_y})")
    assert y_pred_sched.shape == (batch_size, config.forecast_horizon, config.d_y), \
        f"Shape mismatch! Got {y_pred_sched.shape}, expected ({batch_size}, {config.forecast_horizon}, {config.d_y})"
    print("✓ Scheduled sampling test passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_crt_model()

