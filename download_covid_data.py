#!/usr/bin/env python
"""
Standalone script to download and process COVID-19 dataset.

Usage:
    python download_covid_data.py
"""

from crt.config import CRTConfig
from crt.download_covid_dataset import download_and_process_covid_dataset

if __name__ == "__main__":
    # Config for COVID-19 data
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
    
    print("Downloading and processing COVID-19 dataset...")
    download_and_process_covid_dataset(config, output_dir="data/covid")
    print("\nDone! Dataset saved to data/covid/")

