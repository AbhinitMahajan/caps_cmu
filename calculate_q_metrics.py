#!/usr/bin/env python3
"""
Q/Qexp Metric Calculation Using DIRECTLY SAVED Contributions

This version loads the contributions that were saved during training,
avoiding the NNLS approximation and giving TRUE model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.decomposition import NMF
import warnings
import sys
import os

# Add project root to path to import preprocessing module
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preprocessing import load_and_preprocess_data, prepare_data
from src.config import DATA_RAW_DIR

warnings.filterwarnings('ignore')


def load_acsm_data():
    """
    Load and preprocess ACSM data EXACTLY as in training.
    
    Uses the same preprocessing functions from src.data_preprocessing
    to ensure 100% consistency.
    """
    print("Loading ACSM data...")
    
    # Use the EXACT same preprocessing function as training
    df_normalized = load_and_preprocess_data("Spectra_Abhin_reduced.csv")
    
    # Get raw data for uncertainty estimation (before normalization)
    file_path = os.path.join(DATA_RAW_DIR, "Spectra_Abhin_reduced.csv")
    df_raw = pd.read_csv(file_path)
    acsm_features_raw = df_raw.iloc[:, 1:]  # Exclude Time column, same as preprocessing
    X_raw = acsm_features_raw.values
    
    # Convert normalized DataFrame to numpy array (same as prepare_data)
    X_normalized = df_normalized.values
    
    print(f"Data shape: {X_normalized.shape}")
    print(f"  Samples (m): {X_normalized.shape[0]}")
    print(f"  Features (n): {X_normalized.shape[1]} m/z species")
    print(f"  Raw data shape: {X_raw.shape}")
    print(f"  ✓ Using EXACT same preprocessing as training!")
    
    return X_normalized, X_raw


def estimate_uncertainties(X_raw, method='sqrt'):
    """Estimate measurement uncertainties"""
    if method == 'sqrt':
        sigma = np.sqrt(np.abs(X_raw) + 1e-10)
    elif method == 'std':
        std_per_feature = np.std(X_raw, axis=0, keepdims=True)
        sigma = np.tile(std_per_feature, (X_raw.shape[0], 1))
        sigma = np.maximum(sigma, 1e-10)
    elif method == 'constant':
        sigma = 0.1 * np.abs(X_raw) + 1e-10
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    return sigma


def load_experiment_results(exp_dir):
    """
    Load contributions and profiles directly from saved files.
    NO MODEL LOADING NEEDED!
    """
    exp_path = Path(exp_dir)
    
    # Load parameters
    with open(exp_path / "parameters.json", 'r') as f:
        params = json.load(f)
    
    # Load factor profiles
    factor_profiles = np.load(exp_path / "probabilistic_factors.npy")
    
    # Load contributions (THE KEY DIFFERENCE!)
    contributions_path = exp_path / "factor_contributions.npy"
    
    if not contributions_path.exists():
        raise FileNotFoundError(
            f"Contributions not found in {exp_dir}!\n"
            f"Run add_contributions_to_existing_experiments.py first."
        )
    
    contributions = np.load(contributions_path)
    
    return contributions, factor_profiles, params


def calculate_Q(X_true, X_reconstructed, sigma):
    """Calculate Q objective function"""
    residuals = (X_true - X_reconstructed) / sigma
    Q = np.sum(residuals ** 2)
    return Q


def calculate_Qexp(n_samples, n_features, n_factors):
    """Calculate expected Q"""
    Qexp = n_samples * n_features - n_factors * (n_samples + n_features)
    return Qexp


def analyze_single_experiment(exp_dir, X_normalized, sigma):
    """Analyze experiment using SAVED contributions"""
    print(f"\n  Analyzing: {Path(exp_dir).name}")
    
    try:
        # Load contributions and profiles directly
        contributions, factor_profiles, params = load_experiment_results(exp_dir)
        
        n_factors = factor_profiles.shape[0]
        n_samples, n_features = X_normalized.shape
        
        print(f"    Factors: {n_factors}")
        print(f"    Contributions shape: {contributions.shape}")
        print(f"    Profiles shape: {factor_profiles.shape}")
        
        # Reconstruct using saved contributions
        X_reconstructed = contributions @ factor_profiles
        
        # Calculate MSE
        mse = np.mean((X_normalized - X_reconstructed) ** 2)
        
        # Calculate Q
        Q = calculate_Q(X_normalized, X_reconstructed, sigma)
        
        # Calculate Qexp
        Qexp = calculate_Qexp(n_samples, n_features, n_factors)
        
        # Calculate Q/Qexp ratio
        Q_Qexp_ratio = Q / Qexp
        
        print(f"    Q: {Q:.2f}")
        print(f"    Qexp: {Qexp:.2f}")
        print(f"    Q/Qexp: {Q_Qexp_ratio:.6f}")
        print(f"    MSE: {mse:.6f}")
        
        return {
            'n_factors': n_factors,
            'Q': Q,
            'Qexp': Qexp,
            'Q_Qexp_ratio': Q_Qexp_ratio,
            'mse': mse,
            'ortho_weight': params.get('ortho_weight', None),
            'entropy_weight': params.get('entropy_weight_a', None),
            'experiment_name': Path(exp_dir).name,
            'method': 'Direct (saved contributions)'
        }
        
    except FileNotFoundError as e:
        print(f"    ✗ {e}")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def perform_nmf_analysis(X_normalized, X_raw, sigma, max_factors=9):
    """
    Perform NMF for comparison using best practices.
    
    Uses normalized data (same as autoencoder) for fair comparison.
    NMF parameters chosen for stability and reproducibility.
    """
    print("\n" + "="*80)
    print("Performing NMF Analysis for Comparison")
    print("="*80)
    
    results = []
    n_samples, n_features = X_normalized.shape
    
    for n_factors in range(3, max_factors + 1):
        print(f"\n  NMF with {n_factors} factors...")
        
        # NMF with robust initialization and convergence
        nmf = NMF(
            n_components=n_factors,
            init='nndsvda',           # Non-Negative Double SVD (best for sparse data)
            solver='cd',              # Coordinate Descent (faster, numerically stable)
            beta_loss='frobenius',    # Standard Frobenius norm (same as MSE)
            max_iter=1000,            # Sufficient iterations
            random_state=42,          # Reproducibility
            alpha_W=0.0,              # No regularization on W (for fair comparison)
            alpha_H=0.0,              # No regularization on H (for fair comparison)
            l1_ratio=0.0,             # No L1 penalty
            tol=1e-4                  # Convergence tolerance
        )
        
        # Fit NMF
        W = nmf.fit_transform(X_normalized)  # Contributions (n_samples × n_factors)
        H = nmf.components_                   # Profiles (n_factors × n_features)
        
        # Verify NMF converged
        if nmf.n_iter_ >= nmf.max_iter:
            print(f"    ⚠ Warning: NMF did not converge (reached max_iter={nmf.max_iter})")
        
        # Reconstruct using NMF factors
        X_reconstructed = W @ H
        
        # Calculate metrics (SAME as autoencoder)
        mse = np.mean((X_normalized - X_reconstructed) ** 2)
        Q = calculate_Q(X_normalized, X_reconstructed, sigma)
        Qexp = calculate_Qexp(n_samples, n_features, n_factors)
        Q_Qexp_ratio = Q / Qexp
        
        # Additional NMF-specific metrics
        reconstruction_error = nmf.reconstruction_err_  # Built-in Frobenius norm
        
        print(f"    Q: {Q:.2f}")
        print(f"    Qexp: {Qexp:.2f}")
        print(f"    Q/Qexp: {Q_Qexp_ratio:.6f}")
        print(f"    MSE: {mse:.6f}")
        print(f"    NMF reconstruction error: {reconstruction_error:.6f}")
        print(f"    Converged in {nmf.n_iter_} iterations")
        
        results.append({
            'n_factors': n_factors,
            'Q': Q,
            'Qexp': Qexp,
            'Q_Qexp_ratio': Q_Qexp_ratio,
            'mse': mse,
            'nmf_recon_error': reconstruction_error,
            'n_iter': nmf.n_iter_,
            'method': 'NMF'
        })
    
    return results


def plot_comparison(results_dict, output_dir, uncertainty_method):
    """Create comprehensive comparison plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Main figure with 4 subplots (wider to accommodate legend)
    fig, axes = plt.subplots(2, 2, figsize=(22, 13))
    fig.suptitle(f'Q/Qexp Analysis Using DIRECT Contributions\n(Uncertainty: {uncertainty_method})',
                 fontsize=17, fontweight='bold')
    
    # Plot 1: Q/Qexp ratios (THE KEY METRIC)
    ax1 = axes[0, 0]
    # Define colors for different epoch counts
    epoch_colors = {'25': 'darkred', '50': 'crimson', '60': 'orangered', '100': 'chocolate', '250': 'steelblue', '600': 'darkorange', '2000': 'forestgreen', '3000': 'purple'}
    color_idx = 0
    autoencoder_colors = ['darkred', 'crimson', 'orangered', 'chocolate', 'steelblue', 'darkorange', 'forestgreen', 'purple', 'teal']
    
    for config_name, results in results_dict.items():
        # Skip NMF baseline in plots
        if 'NMF' in config_name:
            continue
        if results:
            n_factors_list = [r['n_factors'] for r in results]
            q_qexp_list = [r['Q_Qexp_ratio'] for r in results]
            
            # Extract epochs from config name for color coding
            color = 'gray'
            for epoch, col in epoch_colors.items():
                if epoch in config_name:
                    color = col
                    break
            if color == 'gray' and 'Autoencoder' in config_name:
                color = autoencoder_colors[color_idx % len(autoencoder_colors)]
                color_idx += 1
            
            ax1.plot(n_factors_list, q_qexp_list, 's-', linewidth=2.5, markersize=9,
                    label=config_name, color=color, alpha=0.9)
    
    ax1.axhline(y=1.0, color='red', linestyle=':', linewidth=2.5, label='Q/Qexp = 1 (ideal)', zorder=10)
    ax1.set_xlabel('Number of Factors', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Q/Qexp Ratio', fontsize=13, fontweight='bold')
    ax1.set_title('Q/Qexp vs Number of Factors\n(Elbow point indicates optimal factors)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(3, 10))
    
    # Plot 2: Absolute Q values (UNSCALED - shows true magnitude)
    ax2 = axes[0, 1]
    color_idx = 0
    for config_name, results in results_dict.items():
        # Skip NMF baseline in plots
        if 'NMF' in config_name:
            continue
        if results:
            n_factors_list = [r['n_factors'] for r in results]
            q_list = [r['Q'] for r in results]
            
            color = 'gray'
            for epoch, col in epoch_colors.items():
                if epoch in config_name:
                    color = col
                    break
            if color == 'gray' and 'Autoencoder' in config_name:
                color = autoencoder_colors[color_idx % len(autoencoder_colors)]
                color_idx += 1
            
            ax2.plot(n_factors_list, q_list, 's-', linewidth=2.5, markersize=9,
                    label=config_name, color=color, alpha=0.9)
    
    ax2.set_xlabel('Number of Factors', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Q Value (absolute)', fontsize=13, fontweight='bold')
    ax2.set_title('Absolute Q Values\n(Lower = Better fit, look for elbow)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(range(3, 10))
    
    # Plot 3: Q values on LOG SCALE (better visualization if large differences)
    ax3 = axes[1, 0]
    color_idx = 0
    for config_name, results in results_dict.items():
        # Skip NMF baseline in plots
        if 'NMF' in config_name:
            continue
        if results:
            n_factors_list = [r['n_factors'] for r in results]
            q_list = [r['Q'] for r in results]
            
            color = 'gray'
            for epoch, col in epoch_colors.items():
                if epoch in config_name:
                    color = col
                    break
            if color == 'gray' and 'Autoencoder' in config_name:
                color = autoencoder_colors[color_idx % len(autoencoder_colors)]
                color_idx += 1
            
            ax3.semilogy(n_factors_list, q_list, 's-', linewidth=2.5, markersize=9,
                        label=config_name, color=color, alpha=0.9)
    
    ax3.set_xlabel('Number of Factors', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Q Value (log scale)', fontsize=13, fontweight='bold')
    ax3.set_title('Q Values (Log Scale)\n(Shows relative differences clearly)', 
                  fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', which='both')
    ax3.set_xticks(range(3, 10))
    
    # Plot 4: MSE Reconstruction Error
    ax4 = axes[1, 1]
    color_idx = 0
    for config_name, results in results_dict.items():
        # Skip NMF baseline in plots
        if 'NMF' in config_name:
            continue
        if results:
            n_factors_list = [r['n_factors'] for r in results]
            mse_list = [r['mse'] for r in results]
            
            color = 'gray'
            for epoch, col in epoch_colors.items():
                if epoch in config_name:
                    color = col
                    break
            if color == 'gray' and 'Autoencoder' in config_name:
                color = autoencoder_colors[color_idx % len(autoencoder_colors)]
                color_idx += 1
            
            ax4.plot(n_factors_list, mse_list, 's-', linewidth=2.5, markersize=9,
                    label=config_name, color=color, alpha=0.9)
    
    ax4.set_xlabel('Number of Factors', fontsize=13, fontweight='bold')
    ax4.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax4.set_title('Mean Squared Error\n(Reconstruction quality)', 
                  fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xticks(range(3, 10))
    
    plt.tight_layout()
    save_path = Path(output_dir) / f'q_qexp_direct_{uncertainty_method}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nMain plot saved to: {save_path}")
    plt.close()
    
    # Create a FOCUSED plot with just Q and Q/Qexp side-by-side (wider to accommodate legend)
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 6))
    fig2.suptitle(f'Key Metrics for Factor Selection\n(Direct contributions, uncertainty: {uncertainty_method})',
                  fontsize=16, fontweight='bold')
    
    # Left: Q values
    ax_q = axes2[0]
    color_idx = 0
    for config_name, results in results_dict.items():
        # Skip NMF baseline in plots
        if 'NMF' in config_name:
            continue
        if results:
            n_factors_list = [r['n_factors'] for r in results]
            q_list = [r['Q'] for r in results]
            
            color = 'gray'
            for epoch, col in epoch_colors.items():
                if epoch in config_name:
                    color = col
                    break
            if color == 'gray' and 'Autoencoder' in config_name:
                color = autoencoder_colors[color_idx % len(autoencoder_colors)]
                color_idx += 1
            
            ax_q.plot(n_factors_list, q_list, 's-', linewidth=3, markersize=11,
                     label=config_name, color=color, alpha=0.9)
    
    ax_q.set_xlabel('Number of Factors', fontsize=14, fontweight='bold')
    ax_q.set_ylabel('Q Value', fontsize=14, fontweight='bold')
    ax_q.set_title('Weighted Residual Sum (Q)\nLower = Better Fit', 
                   fontsize=15, fontweight='bold')
    ax_q.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.95)
    ax_q.grid(True, alpha=0.4, linestyle='--')
    ax_q.set_xticks(range(3, 10))
    
    # Right: Q/Qexp ratios
    ax_ratio = axes2[1]
    color_idx = 0
    for config_name, results in results_dict.items():
        # Skip NMF baseline in plots
        if 'NMF' in config_name:
            continue
        if results:
            n_factors_list = [r['n_factors'] for r in results]
            q_qexp_list = [r['Q_Qexp_ratio'] for r in results]
            
            color = 'gray'
            for epoch, col in epoch_colors.items():
                if epoch in config_name:
                    color = col
                    break
            if color == 'gray' and 'Autoencoder' in config_name:
                color = autoencoder_colors[color_idx % len(autoencoder_colors)]
                color_idx += 1
            
            ax_ratio.plot(n_factors_list, q_qexp_list, 's-', linewidth=3, markersize=11,
                         label=config_name, color=color, alpha=0.9)
    
    ax_ratio.axhline(y=1.0, color='red', linestyle=':', linewidth=3, 
                     label='Ideal (Q/Qexp = 1)', zorder=10)
    ax_ratio.set_xlabel('Number of Factors', fontsize=14, fontweight='bold')
    ax_ratio.set_ylabel('Q/Qexp Ratio', fontsize=14, fontweight='bold')
    ax_ratio.set_title('Normalized Fit Quality (Q/Qexp)\nCloser to 1 = Better', 
                       fontsize=15, fontweight='bold')
    ax_ratio.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.95)
    ax_ratio.grid(True, alpha=0.4, linestyle='--')
    ax_ratio.set_xticks(range(3, 10))
    
    plt.tight_layout()
    save_path2 = Path(output_dir) / f'q_comparison_focused_{uncertainty_method}.png'
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Focused plot saved to: {save_path2}")
    plt.close()


def create_summary_table(results_dict, output_dir, uncertainty_method):
    """Create summary CSV"""
    all_results = []
    for config_name, results in results_dict.items():
        for result in results:
            result_copy = result.copy()
            result_copy['configuration'] = config_name
            all_results.append(result_copy)
    
    df = pd.DataFrame(all_results)
    
    csv_path = Path(output_dir) / f'q_qexp_direct_{uncertainty_method}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to: {csv_path}")
    
    return df


def main():
    print("="*80)
    print("Q/Qexp CALCULATION USING DIRECT (SAVED) CONTRIBUTIONS")
    print("="*80)
    print("\nThis version uses contributions saved during training.")
    print("No NNLS approximation - TRUE model performance!\n")
    
    # Load data (using EXACT same preprocessing as training)
    # This ensures X_normalized matches exactly what was used during training
    X_normalized, X_raw = load_acsm_data()
    
    # Verify shapes match (required for proper Q calculation)
    assert X_normalized.shape == X_raw.shape, \
        f"Shape mismatch: normalized {X_normalized.shape} vs raw {X_raw.shape}"
    print(f"✓ Verified: Normalized and raw data shapes match")
    
    # Estimate uncertainties (using raw data, before normalization)
    uncertainty_method = 'sqrt'
    print(f"\nEstimating uncertainties using method: {uncertainty_method}")
    sigma = estimate_uncertainties(X_raw, method=uncertainty_method)
    
    # Output directory
    output_dir = 'analysis_results_q_metrics_direct'
    
    # Find experiment directories
    print("\n" + "="*80)
    print("Finding Experiment Directories")
    print("="*80)
    
    # Specify all experiment directories explicitly
    exp_dirs = [
        Path('experiment_results_20251111_164144'),  # 50 epochs
        Path('experiment_results_20251118_034917'),  # 60 epochs
        Path('experiment_results_20251118_032307'),  # 100 epochs
        Path('experiment_results_20251118_022907'),  # 250 epochs
        Path('experiment_results_20251103_164022'),  # 600 epochs
        Path('experiment_results_20251111_131950')   # 2000 epochs
    ]
    
    # Verify directories exist
    valid_dirs = []
    for exp_dir in exp_dirs:
        if exp_dir.exists():
            valid_dirs.append(exp_dir)
            print(f"✓ Found experiment directory: {exp_dir.name}")
        else:
            print(f"✗ Directory not found: {exp_dir}")
    
    if not valid_dirs:
        print("✗ No experiment directories found!")
        return
    
    # Analyze experiments
    print("\n" + "="*80)
    print("Analyzing Experiments")
    print("="*80)
    
    results_dict = {}
    
    # Process each experiment directory
    for base_dir in valid_dirs:
        print(f"\nAnalyzing experiments from: {base_dir.name}")
        
        # Look for experiments with pattern: factors{N}_temp{T}_ortho{O}_entropy{E}
        found_experiments = []
        for exp_folder in sorted(base_dir.glob('factors*_temp*_ortho*_entropy*')):
            found_experiments.append(exp_folder)
        
        if not found_experiments:
            print(f"  ⚠ No experiment folders found in {base_dir}")
            continue
        
        print(f"  ✓ Found {len(found_experiments)} experiment folders")
        
        # Get epochs from first experiment to label this directory
        first_exp_params_path = found_experiments[0] / "parameters.json"
        epochs = None
        if first_exp_params_path.exists():
            with open(first_exp_params_path, 'r') as f:
                params = json.load(f)
                epochs = params.get('epochs', 'unknown')
        
        # Group by configuration
        configs = {}
        for exp_dir in found_experiments:
            # Parse experiment name: factors{N}_temp{T}_ortho{O}_entropy{E}
            name_parts = exp_dir.name.split('_')
            try:
                factors = int(name_parts[0].replace('factors', ''))
                temp = float(name_parts[1].replace('temp', ''))
                ortho = float(name_parts[2].replace('ortho', ''))
                entropy = float(name_parts[3].replace('entropy', ''))
                
                config_key = (temp, ortho, entropy)
                if config_key not in configs:
                    configs[config_key] = []
                configs[config_key].append((factors, exp_dir))
            except (ValueError, IndexError):
                print(f"  ⚠ Could not parse experiment name: {exp_dir.name}")
                continue
        
        # Analyze each configuration
        for (temp, ortho, entropy), experiments in configs.items():
            config_name = f"Autoencoder ({epochs} epochs, temp={temp}, ortho={ortho}, entropy={entropy})"
            print(f"\n{config_name}")
            print("-" * 80)
            
            results = []
            for factors, exp_dir in sorted(experiments):
                result = analyze_single_experiment(exp_dir, X_normalized, sigma)
                if result:
                    results.append(result)
            
            if results:
                results_dict[config_name] = results
    
    # NMF baseline
    nmf_results = perform_nmf_analysis(X_normalized, X_raw, sigma, max_factors=9)
    results_dict['NMF (baseline)'] = nmf_results
    
    # Create plots and summary
    print("\n" + "="*80)
    print("Generating Plots and Summary")
    print("="*80)
    
    plot_comparison(results_dict, output_dir, uncertainty_method)
    summary_df = create_summary_table(results_dict, output_dir, uncertainty_method)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for config_name in summary_df['configuration'].unique():
        config_df = summary_df[summary_df['configuration'] == config_name]
        if len(config_df) > 0:
            config_df['q_qexp_deviation'] = abs(config_df['Q_Qexp_ratio'] - 1.0)
            optimal_row = config_df.loc[config_df['q_qexp_deviation'].idxmin()]
            
            print(f"\n{config_name}:")
            print(f"  Optimal factors: {optimal_row['n_factors']}")
            print(f"  Q/Qexp: {optimal_row['Q_Qexp_ratio']:.6f}")
            print(f"  MSE: {optimal_row['mse']:.6f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated plots:")
    print(f"  1. q_qexp_direct_{uncertainty_method}.png (4-panel comprehensive view)")
    print(f"  2. q_comparison_focused_{uncertainty_method}.png (Q and Q/Qexp side-by-side)")
    print(f"\nData saved:")
    print(f"  - q_qexp_direct_{uncertainty_method}.csv (all metrics)")
    print("\n✓ Used DIRECT contributions from training")
    print("✓ No NNLS approximation needed")
    print("✓ TRUE model performance with proper Q calculation")
    print("✓ NMF baseline uses same data preprocessing and Q formula")
    print("="*80)


if __name__ == "__main__":
    main()

