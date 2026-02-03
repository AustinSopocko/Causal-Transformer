"""
Synthetic Structural Causal Model (SCM) data generator.

Generates sequences of (x_t, a_t, y_t) following a structural causal model
with temporal dependencies.
"""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SCMParams:
    """Container for SCM parameters that define the data-generating process."""
    B_xa: torch.Tensor  # (d_x, d_a) - effect of actions on state
    W_ax: torch.Tensor  # (d_a, d_x) - effect of state on treatment
    W_aa: torch.Tensor  # (d_a, d_a) - effect of previous treatment
    W_yx: torch.Tensor  # (d_y, d_x) - effect of state on outcome
    W_ya: torch.Tensor  # (d_y, d_a) - effect of treatment on outcome
    W_yy: torch.Tensor  # (d_y, d_y) - effect of previous outcome
    b_a: torch.Tensor   # (d_a,) - treatment bias


def sample_scm_params(
    d_x: int,
    d_a: int,
    d_y: int,
    device: Optional[torch.device] = None,
) -> SCMParams:
    """
    Randomly initialize and return SCMParams.
    
    Args:
        d_x: Dimension of state/covariates
        d_a: Dimension of treatments
        d_y: Dimension of outcomes
        device: Device to create tensors on (default: CPU)
        
    Returns:
        SCMParams with randomly initialized parameters (float32 tensors)
    """
    if device is None:
        device = torch.device("cpu")
    
    # State transition parameters
    B_xa = torch.randn(d_x, d_a, device=device, dtype=torch.float32) * 0.1  # (d_x, d_a)
    
    # Treatment assignment parameters
    W_ax = torch.randn(d_a, d_x, device=device, dtype=torch.float32) * 0.1  # (d_a, d_x)
    W_aa = torch.randn(d_a, d_a, device=device, dtype=torch.float32) * 0.1  # (d_a, d_a)
    b_a = torch.randn(d_a, device=device, dtype=torch.float32) * 0.05  # (d_a,)
    
    # Outcome parameters
    W_yx = torch.randn(d_y, d_x, device=device, dtype=torch.float32) * 0.1  # (d_y, d_x)
    W_ya = torch.randn(d_y, d_a, device=device, dtype=torch.float32) * 0.1  # (d_y, d_a)
    W_yy = torch.randn(d_y, d_y, device=device, dtype=torch.float32) * 0.1  # (d_y, d_y)
    
    return SCMParams(
        B_xa=B_xa,
        W_ax=W_ax,
        W_aa=W_aa,
        W_yx=W_yx,
        W_ya=W_ya,
        W_yy=W_yy,
        b_a=b_a,
    )


def generate_scm_dataset_from_params(
    params: SCMParams,
    n_sequences: int,
    T: int,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
    device: Optional[torch.device] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate synthetic dataset from a Structural Causal Model (SCM) using fixed parameters.
    
    The SCM models temporal dependencies between states (x), treatments (a), and outcomes (y):
    - x_t depends on x_{t-1} and a_{t-1}
    - a_t depends on x_t and a_{t-1} (treatment assignment)
    - y_t depends on x_t, a_t, and y_{t-1}
    
    Args:
        params: SCMParams instance containing fixed SCM parameters
        n_sequences: Number of sequences to generate
        T: Length of each sequence (number of time steps)
        sigma_x: Standard deviation of noise for state transitions (default: 0.1)
        sigma_y: Standard deviation of noise for outcome transitions (default: 0.1)
        device: Device to generate tensors on (default: CPU, or inferred from params)
        
    Returns:
        List of length n_sequences. Each element is a dict:
            {
                "x": Tensor of shape (T, d_x) - states/covariates
                "a": Tensor of shape (T, d_a) - binary treatments
                "y": Tensor of shape (T, d_y) - outcomes
            }
        All tensors are float32.
    """
    if device is None:
        # Infer device from params
        device = params.B_xa.device
    
    # Extract parameters from SCMParams
    B_xa = params.B_xa
    W_ax = params.W_ax
    W_aa = params.W_aa
    W_yx = params.W_yx
    W_ya = params.W_ya
    W_yy = params.W_yy
    b_a = params.b_a
    
    # Get dimensions from params
    d_x = B_xa.shape[0]
    d_a = B_xa.shape[1]
    d_y = W_yx.shape[0]
    
    # Generate sequences
    sequences = []
    
    for seq_idx in range(n_sequences):
        # Initialize tensors for this sequence
        x_seq = torch.zeros(T, d_x, device=device, dtype=torch.float32)  # (T, d_x)
        a_seq = torch.zeros(T, d_a, device=device, dtype=torch.float32)  # (T, d_a)
        y_seq = torch.zeros(T, d_y, device=device, dtype=torch.float32)  # (T, d_y)
        
        # Initial state (t=1)
        # x_1 ~ N(0, 0.5*I) - smaller initial variance
        x_seq[0] = torch.randn(d_x, device=device, dtype=torch.float32) * 0.5
        
        # a_1 ~ Bernoulli(0.5) (element-wise)
        a_seq[0] = torch.bernoulli(torch.full((d_a,), 0.5, device=device, dtype=torch.float32))
        
        # y_1 ~ N(0, 0.5*I) - smaller initial variance
        y_seq[0] = torch.randn(d_y, device=device, dtype=torch.float32) * 0.5
        
        # Generate subsequent time steps (t = 2..T)
        for t in range(1, T):
            # State transition: x_t = 0.7 * x_{t-1} + B_xa @ a_{t-1} + ε_x
            # ε_x ~ N(0, σ_x^2 I)
            epsilon_x = torch.randn(d_x, device=device, dtype=torch.float32) * sigma_x
            x_seq[t] = (
                0.7 * x_seq[t-1] +                    # (d_x,)
                B_xa @ a_seq[t-1] +                    # (d_x, d_a) @ (d_a,) -> (d_x,)
                epsilon_x                              # (d_x,)
            )
            
            # Treatment assignment: logits_a_t = W_ax @ x_t + W_aa @ a_{t-1} + b_a
            logits_a_t = (
                W_ax @ x_seq[t] +                      # (d_a, d_x) @ (d_x,) -> (d_a,)
                W_aa @ a_seq[t-1] +                    # (d_a, d_a) @ (d_a,) -> (d_a,)
                b_a                                    # (d_a,)
            )
            
            # a_t ~ Bernoulli(sigmoid(logits_a_t)) (element-wise)
            probs_a_t = torch.sigmoid(logits_a_t)      # (d_a,)
            a_seq[t] = torch.bernoulli(probs_a_t)     # (d_a,)
            
            # Outcome: y_t = W_yx @ x_t + W_ya @ a_t + W_yy @ y_{t-1} + ε_y
            # ε_y ~ N(0, σ_y^2 I)
            epsilon_y = torch.randn(d_y, device=device, dtype=torch.float32) * sigma_y
            y_seq[t] = (
                W_yx @ x_seq[t] +                       # (d_y, d_x) @ (d_x,) -> (d_y,)
                W_ya @ a_seq[t] +                       # (d_y, d_a) @ (d_a,) -> (d_y,)
                W_yy @ y_seq[t-1] +                     # (d_y, d_y) @ (d_y,) -> (d_y,)
                epsilon_y +                              # (d_y,)
                0.1 * torch.randn(d_y, device=device, dtype=torch.float32)  # Additional controlled noise
            )
        
        # Store sequence
        sequences.append({
            "x": x_seq,  # (T, d_x)
            "a": a_seq,  # (T, d_a)
            "y": y_seq,  # (T, d_y)
        })
    
    # Verify data ranges are reasonable
    all_y_check = torch.cat([seq["y"] for seq in sequences], dim=0)
    y_mean_check = all_y_check.mean().item()
    y_std_check = all_y_check.std().item()
    
    print(f"Generated {n_sequences} sequences of length {T}")
    print(f"  y values - mean: {y_mean_check:.4f}, std: {y_std_check:.4f}")
    print(f"  y value range: [{all_y_check.min().item():.4f}, {all_y_check.max().item():.4f}]")
    
    # Check that std is reasonable (for numerical stability)
    # Lower bound ensures normalization won't be unstable, upper bound prevents extreme values
    assert y_std_check >= 0.1, (
        f"y_std={y_std_check:.4f} is too small (< 0.1) — normalization may be unstable. "
        f"Consider increasing noise levels (sigma_y) or SCM parameter scales."
    )
    if y_std_check > 5.0:
        print(f"WARNING: y_std={y_std_check:.4f} is quite large (> 5.0). This may lead to large RMSE values.")
    
    return sequences


def generate_scm_dataset(
    n_sequences: int,
    T: int,
    d_x: int,
    d_a: int,
    d_y: int,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
    device: Optional[torch.device] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate synthetic dataset from a Structural Causal Model (SCM).
    
    This is a convenience wrapper that samples new SCM parameters each time.
    For consistent training/evaluation, use generate_scm_dataset_from_params() with fixed params.
    
    Args:
        n_sequences: Number of sequences to generate
        T: Length of each sequence (number of time steps)
        d_x: Dimension of state/covariates
        d_a: Dimension of treatments (binary)
        d_y: Dimension of outcomes
        sigma_x: Standard deviation of noise for state transitions (default: 0.1)
        sigma_y: Standard deviation of noise for outcome transitions (default: 0.1)
        device: Device to generate tensors on (default: CPU)
        
    Returns:
        List of length n_sequences. Each element is a dict:
            {
                "x": Tensor of shape (T, d_x) - states/covariates
                "a": Tensor of shape (T, d_a) - binary treatments
                "y": Tensor of shape (T, d_y) - outcomes
            }
        All tensors are float32.
    """
    # Sample new parameters and generate dataset
    params = sample_scm_params(d_x, d_a, d_y, device=device)
    return generate_scm_dataset_from_params(params, n_sequences, T, sigma_x, sigma_y, device)

