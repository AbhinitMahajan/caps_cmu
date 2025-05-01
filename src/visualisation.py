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
