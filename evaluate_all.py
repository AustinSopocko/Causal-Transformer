#!/usr/bin/env python
"""
Evaluate all models and compare results.

Runs evaluation for all model types and prints a comparison table.
"""

import subprocess
import sys
from pathlib import Path


def evaluate_model(model_type: str, data_source: str = "real", data_path: str = "data/covid"):
    """Evaluate a single model and extract RMSE from output."""
    if model_type == "crt":
        checkpoint = "checkpoints/final_model.pt"
        if not Path(checkpoint).exists():
            checkpoint = "checkpoints/final_model_crt.pt"
    else:
        checkpoint = f"checkpoints/final_model_{model_type}.pt"
    
    if not Path(checkpoint).exists():
        print(f"  Checkpoint not found: {checkpoint}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} model...")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "evaluate.py",
        "--data_source", data_source,
        "--data_path", data_path,
        "--model_type", model_type,
        "--checkpoint", checkpoint
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    overall_rmse = None
    baseline_rmse = None
    improvement = None
    
    for line in result.stdout.split('\n'):
        if "Overall RMSE:" in line:
            try:
                overall_rmse = float(line.split("Overall RMSE:")[1].strip())
            except:
                pass
        elif "Baseline (mean) RMSE:" in line:
            try:
                baseline_rmse = float(line.split("Baseline (mean) RMSE:")[1].strip())
            except:
                pass
        elif "Improvement over baseline:" in line:
            try:
                improvement = float(line.split("Improvement over baseline:")[1].split("%")[0].strip())
            except:
                pass
    
    return {
        "model_type": model_type,
        "overall_rmse": overall_rmse,
        "baseline_rmse": baseline_rmse,
        "improvement": improvement
    }


def main():
    """Evaluate all models and print comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all models and compare")
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["synthetic", "real"],
        default="real",
        help="Data source"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/covid",
        help="Path to data directory"
    )
    args = parser.parse_args()
    
    model_types = ["crt", "linear", "mlp", "gru", "tcn"]
    results = []
    
    print("="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)
    
    for model_type in model_types:
        result = evaluate_model(model_type, args.data_source, args.data_path)
        if result:
            results.append(result)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'RMSE':<15} {'Baseline RMSE':<18} {'Improvement':<15}")
    print("-"*60)
    
    def sort_key(r):
        if r['model_type'] == 'crt':
            return (0, r['overall_rmse'] or float('inf'))
        return (1, r['overall_rmse'] or float('inf'))
    
    sorted_results = sorted(results, key=sort_key)
    
    for r in sorted_results:
        model_name = r['model_type'].upper()
        if r['model_type'] == 'crt':
            model_name = "CRT"
        rmse_str = f"{r['overall_rmse']:.6f}" if r['overall_rmse'] is not None else "N/A"
        baseline_str = f"{r['baseline_rmse']:.6f}" if r['baseline_rmse'] is not None else "N/A"
        improvement_str = f"{r['improvement']:.2f}%" if r['improvement'] is not None else "N/A"
        print(f"{model_name:<12} {rmse_str:<15} {baseline_str:<18} {improvement_str:<15}")
    
    print("="*60)
    
    valid_results = [r for r in results if r['overall_rmse'] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['overall_rmse'])
        print(f"\nBest model: {best['model_type'].upper()} (RMSE: {best['overall_rmse']:.6f})")


if __name__ == "__main__":
    main()

