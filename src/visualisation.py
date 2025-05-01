# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys 

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def plot_training_history(history):
    """
    Plots training and validation loss curves from a Keras History object.

    Parameters
    ----------
    history : History
        The Keras History object returned by model.fit(), expected to contain
        keys such as 'loss', 'val_loss', 'deep_output_loss', and 'val_deep_output_loss'.
    """
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('loss', []), label='Train Total Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Total Loss')
    plt.legend()
    plt.show()

    # Plot deep branch (reconstruction) loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('deep_output_loss', []), label='Train Deep Output Loss')
    plt.plot(history.history.get('val_deep_output_loss', []), label='Val Deep Output Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Deep Branch Loss')
    plt.legend()
    plt.show()


def plot_weight_heatmap(weight_matrix, feature_names, factor_labels, figsize=(12, 6), cmap="viridis"):
    """
    Plots a heatmap of the weight matrix.

    Parameters
    ----------
    weight_matrix : numpy.ndarray
        2D array representing weights (e.g., linear decoder kernel).
    feature_names : list of str
        Names of the features along the horizontal axis.
    factor_labels : list of str
        Labels for the rows (e.g., factor names).
    figsize : tuple, optional
        Figure size (default is (12, 6)).
    cmap : str, optional
        Colormap for the heatmap (default is "viridis").
    """
    plt.figure(figsize=figsize)
    sns.heatmap(weight_matrix, annot=True, fmt=".2f",
                cmap=cmap, xticklabels=feature_names, yticklabels=factor_labels)
    plt.xlabel("Features")
    plt.ylabel("Factors")
    plt.title("Weight Matrix Heatmap")
    plt.xticks(rotation=45)
    plt.show()


def plot_bar_chart_for_factor(factor_values, feature_names, factor_label, figsize=(10, 4), color='skyblue'):
    """
    Plots a bar chart for a single factor.

    Parameters
    ----------
    factor_values : numpy.ndarray or list
        The values corresponding to one factor.
    feature_names : list of str
        The names of the features.
    factor_label : str
        Title or label for the factor being plotted.
    figsize : tuple, optional
        Figure size (default is (10, 4)).
    color : str, optional
        Color for the bars (default is 'skyblue').
    """
    plt.figure(figsize=figsize)
    plt.bar(feature_names, factor_values, color=color)
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Contribution")
    plt.title(f"{factor_label} Contribution Profile")
    plt.tight_layout()
    plt.show()


def plot_comparison(nmf_df, ae_df, title_suffix="Comparison"):
    """
    Plots a heatmap and a scatter plot grid to compare the factors between
    two solutions (for example, NMF versus Autoencoder factors).

    Parameters
    ----------
    nmf_df : pandas.DataFrame
        Row-normalized DataFrame for one method (e.g., NMF) where rows are factors.
    ae_df : pandas.DataFrame
        Row-normalized DataFrame for the other method (e.g., AE) with matching columns.
    title_suffix : str, optional
        Suffix for plot titles.
    """
    # Calculate pairwise correlation matrix
    n_factors = nmf_df.shape[0]
    corr_matrix = np.zeros((n_factors, n_factors))
    nmf_factors = nmf_df.index.tolist()
    ae_factors = ae_df.index.tolist()

    for i in range(n_factors):
        for j in range(n_factors):
            nmf_vec = nmf_df.iloc[i].values
            ae_vec = ae_df.iloc[j].values
            corr = np.corrcoef(nmf_vec, ae_vec)[0, 1]
            corr_matrix[i, j] = corr

    # Heatmap of correlations
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=ae_factors, yticklabels=nmf_factors, annot=True, fmt=".2f")
    plt.xlabel("AE Factors")
    plt.ylabel("NMF Factors")
    plt.title(f"Factor Correlation {title_suffix}")
    plt.show()

    # Scatter plot grid
    fig, axes = plt.subplots(nrows=n_factors, ncols=n_factors, figsize=(12, 12))
    for i in range(n_factors):
        for j in range(n_factors):
            ax = axes[i, j]
            nmf_vec = nmf_df.iloc[i].values
            ae_vec = ae_df.iloc[j].values
            ax.scatter(nmf_vec, ae_vec, s=60, alpha=0.7)
            ax.set_title(f"{nmf_factors[i]} vs {ae_factors[j]}")
            ax.set_xlabel(nmf_factors[i])
            ax.set_ylabel(ae_factors[j])
    plt.tight_layout()
    plt.show()


def plot_difference_and_ratio(nmf_df_norm, ae_df_norm, nmf_factor, ae_factor):
    """
    Plots the difference and ratio between two factor profiles.
    
    Parameters
    ----------
    nmf_df_norm : pandas.DataFrame
        Row-normalized DataFrame for method one (e.g., NMF).
    ae_df_norm : pandas.DataFrame
        Row-normalized DataFrame for method two (e.g., AE).
    nmf_factor : str
        Row name of the factor in the nmf_df_norm DataFrame.
    ae_factor : str
        Row name of the factor in the ae_df_norm DataFrame.
    """
    species = nmf_df_norm.columns
    nmf_values = nmf_df_norm.loc[nmf_factor].values
    ae_values = ae_df_norm.loc[ae_factor].values
    difference = nmf_values - ae_values
    # To avoid division by zero, we set a threshold for the denominator.
    ratio = np.where(ae_values > 1e-12, nmf_values / ae_values, np.nan)
    
    # Plot difference
    plt.figure(figsize=(12, 4))
    plt.bar(species, difference, color='skyblue', alpha=0.8)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Species")
    plt.ylabel("Difference (NMF - AE)")
    plt.title(f"Difference: {nmf_factor} - {ae_factor}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plot ratio
    plt.figure(figsize=(12, 4))
    plt.bar(species, ratio, color='salmon', alpha=0.8)
    plt.axhline(1, color='black', linewidth=1, linestyle='--')
    plt.xlabel("Species")
    plt.ylabel("Ratio (NMF ÷ AE)")
    plt.title(f"Ratio: {nmf_factor} / {ae_factor}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
