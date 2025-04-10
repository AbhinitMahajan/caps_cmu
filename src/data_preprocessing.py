# src/data_preprocessing.py
import os
import pandas as pd
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
import numpy as np

def load_and_preprocess_data(filename="Spectra_Abhin_reduced.csv"):
    file_path = os.path.join(DATA_RAW_DIR, filename)
    acsm_data = pd.read_csv(file_path)
    # Exclude 'Time' column and perform min-max normalization
    acsm_features = acsm_data.iloc[:, 1:]
    min_val = acsm_features.min().min()
    max_val = acsm_features.max().max()
    acsm_scaled = (acsm_features - min_val) / (max_val - min_val)
    acsm_scaled_df = pd.DataFrame(acsm_scaled, columns=acsm_features.columns)
    
    # Optionally, save the normalized data
    processed_file = os.path.join(DATA_PROCESSED_DIR, 'normalized_acsm_data.csv')
    acsm_scaled_df.to_csv(processed_file, index=False)
    print(f'Normalization complete. Data saved as {processed_file}')
    return acsm_scaled_df


def prepare_data(df):
    """
    Prepare data for training.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized DataFrame with shape (num_samples, 43).

    Returns
    -------
    X_input : numpy.ndarray
        Input data for the model with shape (num_samples, 43, 1).
    X_target : numpy.ndarray
        Target for reconstruction (the original spectrum) with shape (num_samples, 43).
    """
    X = df.values  # shape: (num_samples, 43)
    X_input = np.expand_dims(X, axis=-1)  # shape: (num_samples, 43, 1)
    X_target = X  # Target is the original spectrum
    return X_input, X_target
