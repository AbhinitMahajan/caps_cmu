"""
Comprehensive analysis of probabilistic factors vs NMF across all experiments.
Analyzes factor correlations, distinctiveness, and optimal matching.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and preprocess the ACSM data using same method as training"""
    df = pd.read_csv('data/raw/Spectra_Abhin_reduced.csv')
    
    feature_cols = [col for col in df.columns if col.startswith('m/Q')]
    X = df[feature_cols].values
    
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    
    min_val = X.min()
    max_val = X.max()
    X_scaled = (X - min_val) / (max_val - min_val)
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {len(feature_cols)} m/z species")
    print(f"Preprocessing: min={min_val:.6f}, max={max_val:.6f}")
    
    return X_scaled, feature_cols


def perform_nmf_decompositions(X, max_components=9):
    """Perform NMF for 3 to max_components factors"""
    nmf_results = {}
    
    print("\nPerforming NMF decompositions...")
    for n in range(3, max_components + 1):
        nmf = NMF(n_components=n, random_state=42, max_iter=1000)
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        H_normalized = H / H.sum(axis=1, keepdims=True)
        
        nmf_results[n] = {
            'W': W,
            'H': H_normalized,
            'model': nmf,
            'reconstruction_error': nmf.reconstruction_err_
        }
        
        print(f"  NMF with {n} factors: reconstruction error = {nmf.reconstruction_err_:.6f}")
    
    return nmf_results


def load_probabilistic_factors_all(base_dir='experiment_result'):
    """Load probabilistic factors from all experiments"""
    prob_results = {}
    
    print("\nLoading probabilistic factors...")
    for n_factors in range(3, 10):
        factor_dirs = list(Path(base_dir).glob(f'factors{n_factors}_temp0.1_entropy*'))
        
        if not factor_dirs:
            print(f"  Warning: No results found for {n_factors} factors")
            continue
        
        all_factors = []
        all_entropy_weights = []
        
        for factor_dir in sorted(factor_dirs):
            factor_path = factor_dir / 'probabilistic_factors.npy'
            if factor_path.exists():
                factors = np.load(factor_path)
                all_factors.append(factors)
                
                entropy_weight = str(factor_dir.name).split('entropy')[1]
                all_entropy_weights.append(float(entropy_weight))
        
        prob_results[n_factors] = {
            'factors': all_factors,
            'entropy_weights': all_entropy_weights,
            'directories': [str(d.name) for d in sorted(factor_dirs)]
        }
        
        print(f"  Loaded {len(all_factors)} experiments for {n_factors} factors")
    
    return prob_results


def compute_cross_correlation_matrix(factors1, factors2):
    """
    Compute correlation matrix between all pairs of factors from two methods.
    Returns matrix of shape (n_factors1, n_factors2) with correlation values.
    """
    n1, n_features1 = factors1.shape
    n2, n_features2 = factors2.shape
    
    assert n_features1 == n_features2, "Feature dimensions must match"
    
    corr_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            corr, _ = pearsonr(factors1[i], factors2[j])
            corr_matrix[i, j] = corr
    
    return corr_matrix


def find_optimal_matching(corr_matrix):
    """
    Find optimal 1-to-1 matching between factors using Hungarian algorithm.
    Returns matched pairs and their correlations.
    """
    cost_matrix = -np.abs(corr_matrix)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            'prob_factor': i,
            'nmf_factor': j,
            'correlation': corr_matrix[i, j]
        })
    
    return matches


def compute_intra_method_correlations(factors):
    """
    Compute average correlation between all pairs of factors within a method.
    Measures factor distinctiveness (lower = more distinct).
    """
    n_factors = factors.shape[0]
    
    if n_factors < 2:
        return 0.0
    
    correlations = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            corr, _ = pearsonr(factors[i], factors[j])
            correlations.append(abs(corr))
    
    return np.mean(correlations), np.std(correlations), correlations


def plot_correlation_matrix(corr_matrix, n_factors, title):
    """Plot correlation matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1,
                xticklabels=[f'NMF {i+1}' for i in range(corr_matrix.shape[1])],
                yticklabels=[f'Prob {i+1}' for i in range(corr_matrix.shape[0])],
                ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('NMF Factors', fontsize=12)
    ax.set_ylabel('Probabilistic Factors', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_matched_factors(prob_factors, nmf_factors, matches, feature_cols, n_factors, entropy_weight):
    """Plot side-by-side comparison of optimally matched factors"""
    n_matches = len(matches)
    
    fig, axes = plt.subplots(n_matches, 2, figsize=(20, 5 * n_matches))
    if n_matches == 1:
        axes = axes.reshape(1, -1)
    
    for idx, match in enumerate(matches):
        prob_idx = match['prob_factor']
        nmf_idx = match['nmf_factor']
        corr = match['correlation']
        
        prob_factor = prob_factors[prob_idx]
        nmf_factor = nmf_factors[nmf_idx]
        
        axes[idx, 0].bar(range(len(feature_cols)), prob_factor, alpha=0.7, color='lightcoral')
        axes[idx, 0].set_title(f'Probabilistic Factor {prob_idx+1} (entropy={entropy_weight})', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('m/z Species', fontsize=10)
        axes[idx, 0].set_ylabel('Probability', fontsize=10)
        axes[idx, 0].set_xticks(range(len(feature_cols)))
        axes[idx, 0].set_xticklabels(feature_cols, rotation=90, fontsize=8)
        axes[idx, 0].grid(axis='y', alpha=0.3)
        
        axes[idx, 1].bar(range(len(feature_cols)), nmf_factor, alpha=0.7, color='skyblue')
        axes[idx, 1].set_title(f'NMF Factor {nmf_idx+1}', fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('m/z Species', fontsize=10)
        axes[idx, 1].set_ylabel('Probability', fontsize=10)
        axes[idx, 1].set_xticks(range(len(feature_cols)))
        axes[idx, 1].set_xticklabels(feature_cols, rotation=90, fontsize=8)
        axes[idx, 1].grid(axis='y', alpha=0.3)
        
        axes[idx, 0].text(0.98, 0.98, f'Correlation: {corr:.4f}',
                          transform=axes[idx, 0].transAxes,
                          ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                          fontsize=11, fontweight='bold')
        
        axes[idx, 1].text(0.98, 0.98, f'Correlation: {corr:.4f}',
                          transform=axes[idx, 1].transAxes,
                          ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                          fontsize=11, fontweight='bold')
    
    fig.suptitle(f'{n_factors} Factors: Optimal Matching between Methods', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.tight_layout()
    return fig


def plot_intra_correlation_trends(nmf_intra_stats, prob_intra_stats):
    """Plot how intra-method correlations change with number of factors"""
    n_factors_list = sorted(nmf_intra_stats.keys())
    
    nmf_means = [nmf_intra_stats[n]['mean'] for n in n_factors_list]
    nmf_stds = [nmf_intra_stats[n]['std'] for n in n_factors_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].errorbar(n_factors_list, nmf_means, yerr=nmf_stds,
                     marker='o', linewidth=2, markersize=8, capsize=5,
                     label='NMF', color='skyblue')
    
    for entropy_weight in prob_intra_stats[n_factors_list[0]].keys():
        prob_means = [prob_intra_stats[n][entropy_weight]['mean'] for n in n_factors_list]
        prob_stds = [prob_intra_stats[n][entropy_weight]['std'] for n in n_factors_list]
        
        axes[0].errorbar(n_factors_list, prob_means, yerr=prob_stds,
                        marker='s', linewidth=2, markersize=8, capsize=5,
                        label=f'Prob (entropy={entropy_weight})', alpha=0.7)
    
    axes[0].set_xlabel('Number of Factors', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Absolute Intra-Method Correlation', fontsize=12, fontweight='bold')
    axes[0].set_title('Factor Distinctiveness: Lower is Better', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(n_factors_list)
    
    best_entropy = min(prob_intra_stats[n_factors_list[0]].keys(),
                       key=lambda x: prob_intra_stats[n_factors_list[-1]][x]['mean'])
    
    prob_best_means = [prob_intra_stats[n][best_entropy]['mean'] for n in n_factors_list]
    
    axes[1].plot(n_factors_list, nmf_means, marker='o', linewidth=3, markersize=10,
                label='NMF', color='skyblue')
    axes[1].plot(n_factors_list, prob_best_means, marker='s', linewidth=3, markersize=10,
                label=f'Probabilistic (best: entropy={best_entropy})', color='lightcoral')
    
    axes[1].fill_between(n_factors_list,
                         np.array(nmf_means) - np.array(nmf_stds),
                         np.array(nmf_means) + np.array(nmf_stds),
                         alpha=0.2, color='skyblue')
    
    axes[1].set_xlabel('Number of Factors', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Intra-Method Correlation', fontsize=12, fontweight='bold')
    axes[1].set_title('Best Configurations Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(n_factors_list)
    
    plt.tight_layout()
    return fig


def analyze_single_configuration(n_factors, prob_factors, nmf_factors, feature_cols, entropy_weight, output_dir):
    """Complete analysis for a single factor configuration"""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {n_factors} Factors (Entropy Weight: {entropy_weight})")
    print(f"{'='*80}")
    
    corr_matrix = compute_cross_correlation_matrix(prob_factors, nmf_factors)
    
    print("\nCross-Correlation Matrix:")
    corr_df = pd.DataFrame(corr_matrix,
                          index=[f'Prob_{i+1}' for i in range(prob_factors.shape[0])],
                          columns=[f'NMF_{i+1}' for i in range(nmf_factors.shape[0])])
    print(corr_df.round(4))
    
    matches = find_optimal_matching(corr_matrix)
    
    print("\nOptimal Factor Matching:")
    print(f"{'Prob Factor':<15} {'NMF Factor':<15} {'Correlation':<15}")
    print("-" * 45)
    for match in matches:
        print(f"{match['prob_factor']+1:<15} {match['nmf_factor']+1:<15} {match['correlation']:<15.4f}")
    
    avg_corr = np.mean([m['correlation'] for m in matches])
    print(f"\nAverage matched correlation: {avg_corr:.4f}")
    
    prob_intra_mean, prob_intra_std, _ = compute_intra_method_correlations(prob_factors)
    nmf_intra_mean, nmf_intra_std, _ = compute_intra_method_correlations(nmf_factors)
    
    print(f"\nIntra-Method Correlations (Factor Distinctiveness):")
    print(f"  Probabilistic: {prob_intra_mean:.4f} ± {prob_intra_std:.4f}")
    print(f"  NMF:           {nmf_intra_mean:.4f} ± {nmf_intra_std:.4f}")
    
    if prob_intra_mean < nmf_intra_mean:
        diff = nmf_intra_mean - prob_intra_mean
        print(f"  Probabilistic factors are MORE distinct (difference: {diff:.4f})")
    else:
        diff = prob_intra_mean - nmf_intra_mean
        print(f"  NMF factors are MORE distinct (difference: {diff:.4f})")
    
    fig1 = plot_correlation_matrix(corr_matrix, n_factors, 
                                   f'{n_factors} Factors Cross-Correlation (entropy={entropy_weight})')
    fig1.savefig(f'{output_dir}/correlation_matrix_{n_factors}factors_entropy{entropy_weight}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = plot_matched_factors(prob_factors, nmf_factors, matches, feature_cols, 
                               n_factors, entropy_weight)
    fig2.savefig(f'{output_dir}/matched_factors_{n_factors}factors_entropy{entropy_weight}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    return {
        'correlation_matrix': corr_matrix,
        'matches': matches,
        'avg_correlation': avg_corr,
        'prob_intra_mean': prob_intra_mean,
        'prob_intra_std': prob_intra_std,
        'nmf_intra_mean': nmf_intra_mean,
        'nmf_intra_std': nmf_intra_std
    }


def main():
    print("="*80)
    print("COMPREHENSIVE FACTOR ANALYSIS: PROBABILISTIC vs NMF")
    print("="*80)
    
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    X_scaled, feature_cols = load_data()
    
    nmf_results = perform_nmf_decompositions(X_scaled, max_components=9)
    
    prob_results = load_probabilistic_factors_all()
    
    all_results = {}
    nmf_intra_stats = {}
    prob_intra_stats = {}
    
    for n_factors in range(3, 10):
        if n_factors not in prob_results:
            print(f"\nSkipping {n_factors} factors: no probabilistic data")
            continue
        
        nmf_factors = nmf_results[n_factors]['H']
        nmf_intra_mean, nmf_intra_std, _ = compute_intra_method_correlations(nmf_factors)
        nmf_intra_stats[n_factors] = {'mean': nmf_intra_mean, 'std': nmf_intra_std}
        
        if n_factors not in prob_intra_stats:
            prob_intra_stats[n_factors] = {}
        
        all_results[n_factors] = {}
        
        for idx, prob_factors in enumerate(prob_results[n_factors]['factors']):
            entropy_weight = prob_results[n_factors]['entropy_weights'][idx]
            
            results = analyze_single_configuration(
                n_factors, prob_factors, nmf_factors, feature_cols,
                entropy_weight, output_dir
            )
            
            all_results[n_factors][entropy_weight] = results
            
            prob_intra_stats[n_factors][entropy_weight] = {
                'mean': results['prob_intra_mean'],
                'std': results['prob_intra_std']
            }
    
    print("\n" + "="*80)
    print("OVERALL TRENDS: FACTOR DISTINCTIVENESS")
    print("="*80)
    
    fig_trends = plot_intra_correlation_trends(nmf_intra_stats, prob_intra_stats)
    fig_trends.savefig(f'{output_dir}/intra_correlation_trends.png', dpi=300, bbox_inches='tight')
    plt.close(fig_trends)
    
    summary_data = []
    for n_factors in sorted(all_results.keys()):
        for entropy_weight in sorted(all_results[n_factors].keys()):
            result = all_results[n_factors][entropy_weight]
            summary_data.append({
                'n_factors': n_factors,
                'entropy_weight': entropy_weight,
                'avg_cross_correlation': result['avg_correlation'],
                'prob_intra_corr': result['prob_intra_mean'],
                'nmf_intra_corr': result['nmf_intra_mean'],
                'distinctiveness_advantage': result['nmf_intra_mean'] - result['prob_intra_mean']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/analysis_summary.csv', index=False)
    
    print("\nSummary Statistics by Number of Factors:")
    print(summary_df.groupby('n_factors')[['avg_cross_correlation', 'prob_intra_corr', 
                                           'nmf_intra_corr', 'distinctiveness_advantage']].mean().round(4))
    
    print("\nBest Configuration by Distinctiveness (lowest intra-correlation):")
    best_config = summary_df.loc[summary_df['prob_intra_corr'].idxmin()]
    print(f"  Factors: {int(best_config['n_factors'])}")
    print(f"  Entropy Weight: {best_config['entropy_weight']}")
    print(f"  Intra-Correlation: {best_config['prob_intra_corr']:.4f}")
    print(f"  Advantage over NMF: {best_config['distinctiveness_advantage']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. Results saved to: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

