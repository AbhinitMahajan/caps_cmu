# src/visualization.py

import os
import sys
import numpy as np
import pandas as pd
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('matplotlib', 'inline')
except Exception:
    pass

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def plot_training_history(history):
    """
    Plots training and validation loss curves from a Keras History object.
    """
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('loss', []), label='Train Total Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Total Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot deep branch (reconstruction) loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('deep_output_loss', []), label='Train Deep Output Loss')
    plt.plot(history.history.get('val_deep_output_loss', []), label='Val Deep Output Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Deep Branch Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weight_heatmap(weight_matrix, feature_names, factor_labels, figsize=(12, 6), cmap="viridis"):
    """
    Plots a heatmap of the weight matrix.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        weight_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=feature_names,
        yticklabels=factor_labels
    )
    plt.xlabel("Features")
    plt.ylabel("Factors")
    plt.title("Linear Decoder Weights Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_bar_chart_for_factor(factor_values, feature_names, factor_label, figsize=(10, 4), color='skyblue'):
    """
    Plots a bar chart for a single factor's weight contributions.
    """
    plt.figure(figsize=figsize)
    plt.bar(feature_names, factor_values, color=color)
    plt.xticks(rotation=90, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Weight")
    plt.title(f"{factor_label} Contribution Profile")
    plt.tight_layout()
    plt.show()


def load_and_plot_linear_weights(
    weights_path,
    data_csv,
    plot_heatmap=True,
    plot_bars=True
):
    """
    Loads linear decoder weights and input feature names, then plots heatmap
    and/or individual factor bar charts.

    Parameters
    ----------
    weights_path : str
        Path to the numpy .npy file containing linear weights (shape: n_factors x n_features).
    data_csv : str
        Path to the raw input CSV file. Column names will be used as feature names.
    plot_heatmap : bool, optional
        Whether to render the heatmap. Default True.
    plot_bars : bool, optional
        Whether to render bar charts for each factor. Default True.
    """
    # Load data
    df = pd.read_csv(data_csv)
    feature_names = df.columns.tolist()

    # Load weights
    weights = np.load(weights_path)
    # weights shape: (n_factors, n_features)
    n_factors, n_features = weights.shape
    if n_features != len(feature_names):
        raise ValueError(
            f"Mismatch: weights have {n_features} features but CSV has {len(feature_names)} columns"
        )

    factor_labels = [f"Factor_{i+1}" for i in range(n_factors)]

    # Plot heatmap
    if plot_heatmap:
        plot_weight_heatmap(weights, feature_names, factor_labels)

    # Plot bar chart per factor
    if plot_bars:
        for i, label in enumerate(factor_labels):
            plot_bar_chart_for_factor(weights[i], feature_names, label)


def plot_comparison(nmf_df, ae_df, title_suffix=""):
    """
    Plot correlation matrix and scatter grid comparing NMF and AE factor profiles.
    
    Parameters
    ----------
    nmf_df : pd.DataFrame
        NMF factor profiles (rows=factors, cols=m/z features)
    ae_df : pd.DataFrame  
        AE factor profiles (rows=factors, cols=m/z features)
    title_suffix : str
        Additional text for plot titles
    """
    n_factors = min(len(nmf_df), len(ae_df))
    
    # Correlation matrix between all factor pairs
    corr_matrix = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        for j in range(n_factors):
            corr_matrix[i, j] = np.corrcoef(nmf_df.iloc[i], ae_df.iloc[j])[0, 1]
    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=[f'AE_F{i+1}' for i in range(n_factors)],
                yticklabels=[f'NMF_F{i+1}' for i in range(n_factors)])
    plt.title(f'Factor Correlation Matrix{title_suffix}')
    plt.xlabel('Autoencoder Factors')
    plt.ylabel('NMF Factors')
    plt.tight_layout()
    plt.show()
    
    # Scatter grid comparing factor profiles
    fig, axes = plt.subplots(n_factors, n_factors, figsize=(12, 12))
    if n_factors == 1:
        axes = np.array([[axes]])
    elif n_factors == 2:
        axes = axes.reshape(2, 2)
        
    for i in range(n_factors):
        for j in range(n_factors):
            ax = axes[i, j]
            ax.scatter(nmf_df.iloc[i], ae_df.iloc[j], alpha=0.6, s=20)
            ax.set_xlabel(f'NMF Factor {i+1}')
            ax.set_ylabel(f'AE Factor {j+1}')
            ax.set_title(f'r={corr_matrix[i,j]:.3f}')
            
            # Add diagonal line
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.suptitle(f'Factor Profile Scatter Grid{title_suffix}')
    plt.tight_layout()
    plt.show()


def plot_difference_and_ratio(nmf_df_norm, ae_df_norm, nmf_factor, ae_factor):
    """
    Plot difference and ratio between selected NMF and AE factors.
    
    Parameters
    ---------- 
    nmf_df_norm : pd.DataFrame
        Row-normalized NMF factor profiles
    ae_df_norm : pd.DataFrame
        Row-normalized AE factor profiles  
    nmf_factor : str
        Name of NMF factor to compare (e.g., "NMF_Factor1")
    ae_factor : str
        Name of AE factor to compare (e.g., "AE_Factor1")
    """
    # Extract factor data
    species = nmf_df_norm.columns
    nmf_values = nmf_df_norm.loc[nmf_factor].values
    ae_values = ae_df_norm.loc[ae_factor].values
    
    # Calculate difference and ratio
    difference = nmf_values - ae_values
    ratio = np.where(ae_values > 1e-12, nmf_values / ae_values, np.nan)
    
    # Plot difference
    plt.figure(figsize=(12, 4))
    plt.bar(species, difference, color='skyblue', alpha=0.8)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Difference: {nmf_factor} - {ae_factor}")
    plt.xlabel("m/z Species")
    plt.ylabel("Difference (NMF - AE)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plot ratio
    plt.figure(figsize=(12, 4))
    plt.bar(species, ratio, color='salmon', alpha=0.8)
    plt.axhline(1, color='black', linewidth=1, linestyle='--')
    plt.title(f"Ratio: {nmf_factor} / {ae_factor}")
    plt.xlabel("m/z Species") 
    plt.ylabel("Ratio (NMF ÷ AE)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Load and visualize linear decoder weights over features."
    )
    parser.add_argument(
        '--data_csv',
        type=str,
        required=True,
        help='Path to raw CSV file containing the input dataset.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=os.path.join(ROOT, 'saved_models', 'linear_weights.npy'),
        help='Path to the .npy file of linear decoder weights.'
    )
    parser.add_argument(
        '--no_heatmap',
        action='store_true',
        help='Skip the heatmap plot.'
    )
    parser.add_argument(
        '--no_bars',
        action='store_true',
        help='Skip the bar charts.'
    )
    args = parser.parse_args()

    load_and_plot_linear_weights(
        weights_path=args.weights,
        data_csv=args.data_csv,
        plot_heatmap=not args.no_heatmap,
        plot_bars=not args.no_bars
    )
