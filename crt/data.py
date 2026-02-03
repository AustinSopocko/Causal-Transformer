import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from .config import CRTConfig, NormalizationStats
from .synthetic import generate_scm_dataset_from_params, sample_scm_params, SCMParams


class TimeSeriesCounterfactualDataset(Dataset):
    """PyTorch Dataset for time series counterfactual data with sliding windows."""
    
    def __init__(
        self, 
        data: List[Dict[str, torch.Tensor]], 
        config: CRTConfig,
        norm_stats: Optional[NormalizationStats] = None
    ):
        """Initialize dataset from list of sequences with optional normalization."""
        self.config = config
        self.norm_stats = norm_stats
        self.samples = []
        
        for seq_dict in data:
            x = seq_dict["x"]
            a = seq_dict["a"]
            y = seq_dict["y"]
            
            T = x.shape[0]
            T_required = config.history_len + config.forecast_horizon
            
            for start_idx in range(T - T_required + 1):
                x_hist = x[start_idx : start_idx + config.history_len]
                a_hist = a[start_idx : start_idx + config.history_len]
                y_hist = y[start_idx : start_idx + config.history_len]
                
                fut_start = start_idx + config.history_len
                fut_end = fut_start + config.forecast_horizon
                a_fut = a[fut_start : fut_end]
                y_fut = y[fut_start : fut_end]
                
                if self.norm_stats is not None:
                    y_hist = (y_hist - self.norm_stats.y_mean) / self.norm_stats.y_std
                    y_fut = (y_fut - self.norm_stats.y_mean) / self.norm_stats.y_std
                
                x_hist = x_hist.unsqueeze(0)
                a_hist = a_hist.unsqueeze(0)
                y_hist = y_hist.unsqueeze(0)
                a_fut = a_fut.unsqueeze(0)
                y_fut = y_fut.unsqueeze(0)
                
                self.samples.append({
                    "x_hist": x_hist,
                    "a_hist": a_hist,
                    "y_hist": y_hist,
                    "a_fut": a_fut,
                    "y_fut": y_fut,
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def make_synthetic_dataset(
    cfg: CRTConfig,
    n_sequences: int,
    T: int,
    norm_stats: Optional[NormalizationStats] = None,
    params: Optional["SCMParams"] = None,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
) -> Tuple[TimeSeriesCounterfactualDataset, "SCMParams"]:
    """Create synthetic SCM dataset. Returns (dataset, scm_params)."""
    # Assert that sequence length is sufficient for sliding windows
    assert cfg.history_len + cfg.forecast_horizon <= T, (
        f"Sequence length T={T} must be at least history_len + forecast_horizon = "
        f"{cfg.history_len} + {cfg.forecast_horizon} = {cfg.history_len + cfg.forecast_horizon}"
    )
    
    # Sample SCM parameters if not provided
    if params is None:
        params = sample_scm_params(cfg.d_x, cfg.d_a, cfg.d_y)
    
    # Generate synthetic SCM sequences using fixed parameters
    # Each sequence has shape: {"x": (T, d_x), "a": (T, d_a), "y": (T, d_y)}
    synthetic_data = generate_scm_dataset_from_params(
        params=params,
        n_sequences=n_sequences,
        T=T,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )
    
    # Wrap in TimeSeriesCounterfactualDataset
    # This creates sliding windows of size (history_len + forecast_horizon)
    dataset = TimeSeriesCounterfactualDataset(synthetic_data, cfg, norm_stats=norm_stats)
    
    return dataset, params


def compute_normalization_stats(data: List[Dict[str, torch.Tensor]]) -> NormalizationStats:
    """Compute normalization stats (mean, std) for y values across dataset."""
    # Collect all y values (before windowing)
    all_y = []
    for seq_dict in data:
        y = seq_dict["y"]  # (T, d_y)
        all_y.append(y)
    
    # Concatenate all y values: (total_T, d_y)
    all_y_tensor = torch.cat(all_y, dim=0)  # (total_T, d_y)
    
    # Compute mean and std over time dimension (across all sequences and time steps)
    # Result: per-feature statistics
    y_mean = torch.mean(all_y_tensor, dim=0)  # (d_y,) - mean per feature dimension
    y_std = torch.std(all_y_tensor, dim=0)    # (d_y,) - std per feature dimension
    
    # Debug: print raw statistics
    raw_y_mean = all_y_tensor.mean().item()
    raw_y_std = all_y_tensor.std().item()
    print(f"TRAIN raw y mean/std (overall): {raw_y_mean:.6f}, {raw_y_std:.6f}")
    print(f"TRAIN raw y mean/std (per feature): mean range [{y_mean.min().item():.4f}, {y_mean.max().item():.4f}], "
          f"std range [{y_std.min().item():.4f}, {y_std.max().item():.4f}]")
    
    # Avoid division by zero: set minimum std to 1e-6
    y_std = torch.clamp(y_std, min=1e-6)
    
    # Assertion: ensure std is not too small
    assert y_std.min() > 1e-6, f"y_std too small — normalization unstable. Min std: {y_std.min().item():.2e}"
    
    # Check normalized values (sample check)
    y_norm_sample = (all_y_tensor[:100] - y_mean) / y_std  # Sample first 100 values
    y_norm_mean = y_norm_sample.mean().item()
    y_norm_std = y_norm_sample.std().item()
    print(f"TRAIN normalized y mean/std (sample): {y_norm_mean:.6f}, {y_norm_std:.6f}")
    
    return NormalizationStats(y_mean=y_mean, y_std=y_std)

