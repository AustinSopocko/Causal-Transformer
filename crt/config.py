from dataclasses import dataclass
import torch


@dataclass
class NormalizationStats:
    """Statistics for normalizing outcome values (y)."""
    
    y_mean: torch.Tensor
    """Mean of all y values in the dataset."""
    
    y_std: torch.Tensor
    """Standard deviation of all y values in the dataset."""


@dataclass
class CRTConfig:
    """Configuration class for CRT (Causal Rollout Transformer) model."""
    
    d_x: int = 128
    """Dimension of input features."""
    
    d_a: int = 64
    """Dimension of action/auxiliary features."""
    
    d_y: int = 128
    """Dimension of output features."""
    
    d_model: int = 256
    """Model dimension (embedding size)."""
    
    n_heads: int = 8
    """Number of attention heads."""
    
    n_layers_enc: int = 4
    """Number of encoder layers."""
    
    n_layers_dec: int = 4
    """Number of decoder layers."""
    
    history_len: int = 10
    """Length of input history sequence."""
    
    forecast_horizon: int = 5
    """Length of forecast horizon."""
    
    dropout: float = 0.1
    """Dropout rate."""
    
    lr: float = 1e-4
    """Learning rate."""
    
    teacher_forcing_start: float = 1.0
    """Initial teacher forcing probability."""
    
    teacher_forcing_end: float = 0.0
    """Final teacher forcing probability after decay."""

