"""
Real-world dataset loading utilities.

Loads data from NumPy files in a directory structure.
"""

import numpy as np
import torch
from typing import List, Dict
from pathlib import Path

from .config import CRTConfig


def load_real_dataset(config: CRTConfig, data_path: str) -> List[Dict[str, torch.Tensor]]:
    """
    Load real-world dataset from NumPy files.
    
    Expected directory structure:
        data_path/
            states.npy   - shape (N, T, d_x)
            actions.npy - shape (N, T, d_a)
            outcomes.npy - shape (N, T, d_y)
    
    Args:
        config: CRTConfig instance with d_x, d_a, d_y dimensions
        data_path: Path to directory containing the .npy files
        
    Returns:
        List of length N, where each element is a dict:
            {
                "x": torch.Tensor of shape (T, d_x),
                "a": torch.Tensor of shape (T, d_a),
                "y": torch.Tensor of shape (T, d_y),
            }
    """
    data_path = Path(data_path)
    
    # Load NumPy arrays
    states_path = data_path / "states.npy"
    actions_path = data_path / "actions.npy"
    outcomes_path = data_path / "outcomes.npy"
    
    if not states_path.exists():
        raise FileNotFoundError(f"states.npy not found in {data_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"actions.npy not found in {data_path}")
    if not outcomes_path.exists():
        raise FileNotFoundError(f"outcomes.npy not found in {data_path}")
    
    states = np.load(states_path)  # (N, T, d_x)
    actions = np.load(actions_path)  # (N, T, d_a)
    outcomes = np.load(outcomes_path)  # (N, T, d_y)
    
    # Verify shapes
    N, T, d_x = states.shape
    N_a, T_a, d_a = actions.shape
    N_o, T_o, d_y = outcomes.shape
    
    # Assert all sequences have same length
    assert N == N_a == N_o, (
        f"Number of sequences mismatch: states={N}, actions={N_a}, outcomes={N_o}"
    )
    assert T == T_a == T_o, (
        f"Sequence length mismatch: states={T}, actions={T_a}, outcomes={T_o}"
    )
    
    # Assert dimensions match config
    assert d_x == config.d_x, (
        f"State dimension mismatch: data has d_x={d_x}, config expects d_x={config.d_x}"
    )
    assert d_a == config.d_a, (
        f"Action dimension mismatch: data has d_a={d_a}, config expects d_a={config.d_a}"
    )
    assert d_y == config.d_y, (
        f"Outcome dimension mismatch: data has d_y={d_y}, config expects d_y={config.d_y}"
    )
    
    # Create list of dicts
    sequences = []
    for i in range(N):
        sequences.append({
            "x": torch.tensor(states[i], dtype=torch.float32),  # (T, d_x)
            "a": torch.tensor(actions[i], dtype=torch.float32),  # (T, d_a)
            "y": torch.tensor(outcomes[i], dtype=torch.float32),  # (T, d_y)
        })
    
    print(f"Loaded {N} sequences of length {T} from {data_path}")
    print(f"  States: shape ({N}, {T}, {d_x})")
    print(f"  Actions: shape ({N}, {T}, {d_a})")
    print(f"  Outcomes: shape ({N}, {T}, {d_y})")
    
    return sequences

