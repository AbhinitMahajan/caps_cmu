# src/training.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import os

# Import data preprocessing and model wrapper
from src.data_preprocessing import load_and_preprocess_data
from src.models import AutoencoderModel
from src.data_preprocessing import prepare_data
from src import config
from src.visualisation import plot_training_history

def get_input(prompt, cast_type, default):
    """
    Helper function to prompt for input with a default value.
    If the input is blank or cannot be cast, returns the default.
    """
    user_input = input(f"{prompt} (Default: {default}): ").strip()
    if not user_input:
        return default
    try:
        return cast_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default



def main():
    print("=== Autoencoder Training Script ===\n")
    
    # ---------------- Hyperparameter and File Path Prompts ----------------
    file_name = input("Enter the file name located in the raw data folder [Spectra_Abhin_reduced.csv]: ").strip() or "Spectra_Abhin_reduced.csv"
    epochs = get_input("Enter number of epochs", int, 400)
    batch_size = get_input("Enter batch size", int, 512)
    n_clusters = get_input("Enter number of clusters", int, 3)
    lambda1 = get_input("Enter lambda1 (consistency loss weight)", float, 0.5)
    lambda2 = get_input("Enter lambda2 (correlation loss weight)", float, 0.6)
    learning_rate = get_input("Enter learning rate", float, 1e-4)
    # New prompts for regularization parameters
    linear_l1 = get_input("Enter linear l1 regularization", float, 1e-5)
    linear_l2 = get_input("Enter linear l2 regularization", float, 1e-3)
    
    # ---------------- Data Preparation ----------------
    df = load_and_preprocess_data(file_name)
    X_input, X_target = prepare_data(df)
    input_dim = X_input.shape[1]  # 43 features
    print(f"For reference Input shape: {X_input.shape}") # (num_samples, 43, 1)

    # ---------------- Build and Compile Model ----------------
    ae_model = AutoencoderModel(
        n_clusters=n_clusters,
        input_dim=input_dim, 
        input_shape=(input_dim, 1),
        lambda1=lambda1,
        lambda2=lambda2,
        linear_l1= linear_l1,  # L1 regularization for the linear output
        linear_l2=linear_l2  # L1 regularization for the linear output
        # You can also pass custom linear regularization parameters here if needed.
    )
    
    ae_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={"deep_output": "mse", "linear_output": "mse"},
        loss_weights={"deep_output": 1.0, "linear_output": 0.0}
    )
    
    print("\nModel Summary:")
    ae_model.summary()
    
    # ---------------- Train the Model ----------------
    history = ae_model.fit(
        X_input,
        {"deep_output": X_target, "linear_output": X_target},
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.25
    )
    
    # ---------------- Plot Training History ----------------
    plot_training_history(history)




    # ---------------- Save Model and Weights ----------------
    # Define directories for saving the model and weights
    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)

    # Save the complete autoencoder model
    model_save_path = os.path.join(model_dir, "autoencoder_model.h5")
    ae_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Retrieve the linear branch layer from the dual decoder and save its weights
    linear_layer = ae_model.dual_decoder.get_layer('linear_output')
    W = linear_layer.get_weights()[0]  # Extract kernel weights
    weights_save_path = os.path.join(model_dir, "linear_weights.npy")
    np.save(weights_save_path, W)
    print(f"Linear weights saved to {weights_save_path}")


if __name__ == "__main__":
    main()
