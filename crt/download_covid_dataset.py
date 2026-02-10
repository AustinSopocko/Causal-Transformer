"""
Download and preprocess COVID-19 hospitalization dataset.

Downloads data from MoH Malaysia COVID-19 repository and converts it to
the format expected by CRT model.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple
import urllib.request
from .config import CRTConfig


def download_covid_data(url: str, output_path: Path) -> pd.DataFrame:
    """Download COVID-19 CSV data from URL."""
    print(f"Downloading COVID-19 data from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
    
    df = pd.read_csv(output_path)
    print(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    return df


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract states, treatments, and outcomes from COVID-19 dataset.
    
    States (x): beds, beds_covid, beds_noncrit, admitted_total, discharged_total
    Treatments (a): mobility_index (synthetic), vaccination_rate (synthetic)
    Outcomes (y): hosp_covid, hosp_pui
    
    Args:
        df: DataFrame with COVID-19 data
        
    Returns:
        Tuple of (states, treatments, outcomes) as numpy arrays
    """
    # Extract state features (x) - map to available columns
    # Using: beds, beds_covid, beds_noncrit, admitted_total, discharged_total
    state_cols = []
    if 'beds' in df.columns:
        state_cols.append('beds')
    if 'beds_covid' in df.columns:
        state_cols.append('beds_covid')
    if 'beds_noncrit' in df.columns:
        state_cols.append('beds_noncrit')
    if 'admitted_total' in df.columns:
        state_cols.append('admitted_total')
    elif 'admitted_covid' in df.columns:
        state_cols.append('admitted_covid')
    if 'discharged_total' in df.columns:
        state_cols.append('discharged_total')
    elif 'discharged_covid' in df.columns:
        state_cols.append('discharged_covid')
    
    # Pad with alternatives if needed
    while len(state_cols) < 5:
        if 'admitted_pui' not in state_cols and 'admitted_pui' in df.columns:
            state_cols.append('admitted_pui')
        elif 'discharged_pui' not in state_cols and 'discharged_pui' in df.columns:
            state_cols.append('discharged_pui')
        else:
            break
    
    if len(state_cols) == 0:
        raise ValueError(f"No suitable state columns found. Available columns: {df.columns.tolist()}")
    
    print(f"Using state columns: {state_cols}")
    
    # Extract outcome features (y)
    outcome_cols = []
    if 'hosp_covid' in df.columns:
        outcome_cols.append('hosp_covid')
    if 'hosp_pui' in df.columns:
        outcome_cols.append('hosp_pui')
    elif 'admitted_covid' in df.columns:
        outcome_cols.append('admitted_covid')
    
    if len(outcome_cols) == 0:
        raise ValueError(f"No suitable outcome columns found. Available columns: {df.columns.tolist()}")
    
    print(f"Using outcome columns: {outcome_cols}")
    
    # Fill missing values with forward fill, then backward fill, then 0
    df_processed = df[state_cols + outcome_cols].copy()
    df_processed = df_processed.ffill().bfill().fillna(0)
    
    # Extract states
    states = df_processed[state_cols].values  # (T, d_x)
    
    # Extract outcomes
    outcomes = df_processed[outcome_cols].values  # (T, d_y)
    
    # Create synthetic treatments
    # mobility_index: synthetic based on admitted_total
    if 'admitted_total' in df_processed.columns:
        admissions = df_processed['admitted_total'].values
        if admissions.max() > 0:
            admissions_normalized = admissions / admissions.max()
        else:
            admissions_normalized = np.zeros_like(admissions)
        mobility_index = 1.0 - admissions_normalized
    elif 'admitted_covid' in df_processed.columns:
        admissions = df_processed['admitted_covid'].values
        if admissions.max() > 0:
            admissions_normalized = admissions / admissions.max()
        else:
            admissions_normalized = np.zeros_like(admissions)
        mobility_index = 1.0 - admissions_normalized
    else:
        # Random mobility if no admission data available
        mobility_index = np.random.uniform(0.3, 1.0, size=len(df_processed))
    
    # vaccination_rate: synthetic, increasing over time with some noise
    T = len(df_processed)
    time_progress = np.linspace(0, 1, T)
    vaccination_rate = time_progress + np.random.normal(0, 0.1, size=T)
    vaccination_rate = np.clip(vaccination_rate, 0.0, 1.0)
    
    treatments = np.column_stack([mobility_index, vaccination_rate])  # (T, 2)
    
    # Add batch dimension: (1, T, d_*)
    states = states[np.newaxis, :, :]  # (1, T, d_x)
    treatments = treatments[np.newaxis, :, :]  # (1, T, d_a)
    outcomes = outcomes[np.newaxis, :, :]  # (1, T, d_y)
    
    print(f"Extracted features:")
    print(f"  States: shape {states.shape}, columns: {state_cols}")
    print(f"  Treatments: shape {treatments.shape}, columns: ['mobility_index', 'vaccination_rate']")
    print(f"  Outcomes: shape {outcomes.shape}, columns: {outcome_cols}")
    
    return states, treatments, outcomes


def create_sequences(
    states: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    config: CRTConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences using sliding windows.
    
    Each sequence has length T = history_len + forecast_horizon.
    Uses sliding windows to create multiple sequences from the time series.
    
    Args:
        states: (1, T_total, d_x) array - full time series
        treatments: (1, T_total, d_a) array - full time series
        outcomes: (1, T_total, d_y) array - full time series
        config: CRTConfig with history_len and forecast_horizon
        
    Returns:
        Tuple of (states_seq, treatments_seq, outcomes_seq) with shapes (N, T, d_*)
        where T = history_len + forecast_horizon
    """
    T_total = states.shape[1]
    T = config.history_len + config.forecast_horizon  # Sequence length
    
    if T_total < T:
        raise ValueError(
            f"Total time series length T_total={T_total} is less than required sequence length "
            f"T={T} (history_len={config.history_len} + forecast_horizon={config.forecast_horizon})"
        )
    
    N = T_total - T + 1
    
    states_seq = np.zeros((N, T, states.shape[2]))
    treatments_seq = np.zeros((N, T, treatments.shape[2]))
    outcomes_seq = np.zeros((N, T, outcomes.shape[2]))
    
    for i in range(N):
        end_idx = i + T
        states_seq[i] = states[0, i:end_idx]
        treatments_seq[i] = treatments[0, i:end_idx]
        outcomes_seq[i] = outcomes[0, i:end_idx]
    
    print(f"Created {N} sequences using sliding windows (sequence_length={T})")
    print(f"  States sequences: shape {states_seq.shape}")
    print(f"  Treatments sequences: shape {treatments_seq.shape}")
    print(f"  Outcomes sequences: shape {outcomes_seq.shape}")
    
    return states_seq, treatments_seq, outcomes_seq


def save_dataset(
    states: np.ndarray,
    treatments: np.ndarray,
    outcomes: np.ndarray,
    output_dir: Path
):
    """Save arrays as NumPy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    states_path = output_dir / "states.npy"
    treatments_path = output_dir / "actions.npy"
    outcomes_path = output_dir / "outcomes.npy"
    
    np.save(states_path, states)
    np.save(treatments_path, treatments)
    np.save(outcomes_path, outcomes)
    
    print(f"\nSaved dataset to {output_dir}:")
    print(f"  {states_path} - shape {states.shape}")
    print(f"  {treatments_path} - shape {treatments.shape}")
    print(f"  {outcomes_path} - shape {outcomes.shape}")


def download_and_process_covid_dataset(
    config: CRTConfig,
    output_dir: str = "data/covid",
    url: str = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/hospital.csv"
):
    """
    Download and process COVID-19 dataset.
    
    Args:
        config: CRTConfig instance
        output_dir: Directory to save processed data
        url: URL to download CSV from
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data
    csv_path = output_dir / "hospital.csv"
    df = download_covid_data(url, csv_path)
    
    # Extract features
    states, treatments, outcomes = extract_features(df)
    
    # Create sequences using sliding windows
    states_seq, treatments_seq, outcomes_seq = create_sequences(
        states, treatments, outcomes, config
    )
    
    # Save as NumPy files
    save_dataset(states_seq, treatments_seq, outcomes_seq, output_dir)
    
    print(f"\nDataset processing complete!")
    print(f"Config dimensions: d_x={config.d_x}, d_a={config.d_a}, d_y={config.d_y}")
    print(f"Actual dimensions: d_x={states_seq.shape[2]}, d_a={treatments_seq.shape[2]}, d_y={outcomes_seq.shape[2]}")
    
    # Verify dimensions match config
    if states_seq.shape[2] != config.d_x:
        print(f"WARNING: State dimension mismatch! Config expects {config.d_x}, data has {states_seq.shape[2]}")
    if treatments_seq.shape[2] != config.d_a:
        print(f"WARNING: Action dimension mismatch! Config expects {config.d_a}, data has {treatments_seq.shape[2]}")
    if outcomes_seq.shape[2] != config.d_y:
        print(f"WARNING: Outcome dimension mismatch! Config expects {config.d_y}, data has {outcomes_seq.shape[2]}")


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path to import config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crt.config import CRTConfig
    
    config = CRTConfig(
        d_x=5,  # beds_icu, beds_hosp, beds_ventilator, cases_new, tests_new
        d_a=2,  # mobility_index, vaccination_rate
        d_y=2,  # hosp, icu
        d_model=64,
        n_heads=4,
        n_layers_enc=3,
        n_layers_dec=3,
        history_len=10,
        forecast_horizon=5,
        dropout=0.1,
        lr=1e-4,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.0
    )
    
    download_and_process_covid_dataset(config)

