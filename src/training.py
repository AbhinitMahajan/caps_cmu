# src/training.py

import numpy as np
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# CRITICAL: Import and apply seed configuration FIRST
from src import config
# Ensure the config module's seed settings are applied
import random
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = str(config.SEED_VALUE)
random.seed(config.SEED_VALUE)
np.random.seed(config.SEED_VALUE)
tf.random.set_seed(config.SEED_VALUE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Import data preprocessing and model wrapper
from src.data_preprocessing import load_and_preprocess_data
from src.models import AutoencoderModel
from src.data_preprocessing import prepare_data
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


def ensure_reproducibility():
    """
    Re-apply all seed settings to ensure reproducibility.
    Call this before model creation and training.
    """
    print("Ensuring reproducibility with seed:", config.SEED_VALUE)
    os.environ['PYTHONHASHSEED'] = str(config.SEED_VALUE)
    random.seed(config.SEED_VALUE)
    np.random.seed(config.SEED_VALUE)
    tf.random.set_seed(config.SEED_VALUE)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Additional TensorFlow determinism settings
    tf.config.experimental.enable_op_determinism()


def main():
    print("=== Autoencoder Training Script ===\n")
    
    # ---------------- GPU Diagnostics & Config ----------------
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Detected GPUs: {gpus}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Warning: could not enable memory growth on {gpu}: {e}")
    except Exception as e:
        print(f"GPU query failed: {e}")
    
    # ---------------- Hyperparameter and File Path Prompts ----------------
    file_name = input("Enter the file name located in the raw data folder [Spectra_Abhin_reduced.csv]: ").strip() or "Spectra_Abhin_reduced.csv"
    epochs = get_input("Enter number of epochs", int, 400)
    batch_size = get_input("Enter batch size", int, 512)
    n_clusters = get_input("Enter number of clusters", int, 3)
    lambda1 = get_input("Enter lambda1 (consistency loss weight)", float, 0.5)
    lambda2 = get_input("Enter lambda2 (correlation loss weight)", float, 0.6)
    learning_rate = get_input("Enter learning rate", float, 1e-4)
    # Regularization parameters
    linear_l1 = get_input("Enter linear l1 regularization", float, 1e-5)
    linear_l2 = get_input("Enter linear l2 regularization", float, 1e-3)
    # Probabilistic factor parameters
    temperature = get_input("Enter temperature for probabilistic factors (lower=sharper, higher=smoother)", float, 1.0)
    ortho_weight = get_input("Enter orthogonality weight for factor profiles (higher=more diverse factors)", float, 1e-2)
    
    # ---------------- Data Preparation ----------------
    df = load_and_preprocess_data(file_name)
    X_input, X_target = prepare_data(df)
    input_dim = X_input.shape[1]  # e.g. 43 features
    print(f"For reference Input shape: {X_input.shape}")  # (num_samples, 43, 1)
    
    # Create deterministic train/validation split to ensure reproducibility
    n_samples = X_input.shape[0]
    split_idx = int(0.75 * n_samples)  # 75% train, 25% validation
    
    X_train = X_input[:split_idx]
    X_val = X_input[split_idx:]
    y_train = X_target[:split_idx] 
    y_val = X_target[split_idx:]
    
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # ---------------- Ensure Reproducibility Before Model Creation ----------------
    ensure_reproducibility()

    # ---------------- Build and Compile Model ----------------
    # Prefer GPU if available
    physical_gpus = tf.config.list_physical_devices('GPU')
    device_scope = '/GPU:0' if physical_gpus else '/CPU:0'
    print(f"Using device scope: {device_scope}")
    with tf.device(device_scope):
        ae_model = AutoencoderModel(
            n_clusters=n_clusters,
            input_shape=(input_dim, 1),
            lambda1=lambda1,
            lambda2=lambda2,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            temperature=temperature,
            ortho_weight=ortho_weight
        )

    # Create optimizer with deterministic behavior
    optimizer = Adam(learning_rate=learning_rate)
    
    ae_model.compile(
        optimizer=optimizer,
        loss={"deep_output": "mse", "linear_output": "mse"},
        loss_weights={"deep_output": 1.0, "linear_output": 0.0}
    )

    # ---------------- Sanity-check I/O ----------------
    for inp in ae_model.model.inputs:
        print("Model input:", inp.name, inp.shape)
    for out in ae_model.model.outputs:
        print("Model output:", out.name, out.shape)

    print("Inspecting model I/O:")
    print(" Inputs:", ae_model.model.inputs)
    print(" Outputs:", ae_model.model.outputs)

    print("ae_model.model object:", ae_model.model)
    print("   .name:", ae_model.model.name)
    print("   .input_shape:", ae_model.model.input_shape)
    print("   .output_shape:", ae_model.model.output_shape)
    print("X_input[:1].shape:", X_input[:1].shape)

    sample = ae_model.model(X_input[:1])
    print(" Sample deep_output shape:", sample['deep_output'].shape)
    print(" Sample linear_output shape:", sample['linear_output'].shape)

    print("\nModel Summary:")
    ae_model.summary()
    
    # ---------------- Train the Model ----------------
    print("Training X_input shape:", X_input.shape)
    print("Training X_target shape for deep:", X_target.shape)
    print("Running a manual forward pass through the model for 1 batch…")
    _ = ae_model.model(X_input[:batch_size])
    print("Manual forward pass succeeded.")

    with tf.device(device_scope):
        history = ae_model.fit(
            X_train,
            {"deep_output": y_train, "linear_output": y_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, {"deep_output": y_val, "linear_output": y_val}),
            shuffle=False  # Disable shuffling for reproducibility
        )
    
    # ---------------- Plot Training History ----------------
    plot_training_history(history)

    # ---------------- Save Model and Weights ----------------
    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(model_dir, "autoencoder_model.h5")
    ae_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save probabilistic factor profiles (interpretable weights)
    probabilistic_layer = ae_model.model.get_layer('probabilistic_factors')
    factor_logits_layer = probabilistic_layer.factor_logits
    W_logits = factor_logits_layer.get_weights()[0]  # Raw logits weights
    
    # Save raw logits weights (for model reconstruction)
    logits_save_path = os.path.join(model_dir, "factor_logits_weights.npy")
    np.save(logits_save_path, W_logits)
    print(f"Factor logits weights saved to {logits_save_path}")
    
    # Generate and save probabilistic factor profiles using REAL encoder outputs
    # This approach uses actual learned latent representations instead of artificial one-hot vectors
    
    # Extract the encoder part of the model
    encoder = tf.keras.Model(
        inputs=ae_model.model.input,
        outputs=ae_model.model.get_layer('latent').output,
        name='encoder'
    )
    
    # Get real latent vectors from training data
    print("Generating factor profiles using real encoder outputs...")
    real_latent_vectors = encoder.predict(X_train, verbose=0)
    print(f"Real latent vectors shape: {real_latent_vectors.shape}")
    
    # Find representative samples for each factor (representative-sample method only)
    n_factors = n_clusters
    factor_profiles = []
    
    print(f"\nFinding representative samples for {n_factors} factors...")
    
    for i in range(n_factors):
        # Find the sample with highest activation for factor i
        factor_activations = real_latent_vectors[:, i]
        best_sample_idx = np.argmax(factor_activations)
        representative_latent = real_latent_vectors[best_sample_idx:best_sample_idx+1]
        
        print(f"Factor {i+1}: Using sample {best_sample_idx} with activation {factor_activations[best_sample_idx]:.4f}")
        print(f"  Representative latent vector: {representative_latent[0]}")
        
        # Get the factor profile for this representative latent vector
        factor_profile = probabilistic_layer(representative_latent).numpy()
        factor_profiles.append(factor_profile[0])  # Remove batch dimension
    
    # Alternative Method: Create "enhanced one-hot" vectors based on real latent patterns
    print(f"\nAlternative: Creating enhanced factor profiles...")
    enhanced_factor_profiles = []
    
    for i in range(n_factors):
        # Create a vector that emphasizes factor i but uses real latent patterns
        # Take the mean latent vector and enhance factor i
        mean_latent = np.mean(real_latent_vectors, axis=0)
        enhanced_latent = mean_latent.copy()
        enhanced_latent[i] = np.max(real_latent_vectors[:, i])  # Use max activation for this factor
        
        # Normalize to maintain realistic proportions
        enhanced_latent = enhanced_latent / np.sum(enhanced_latent)
        
        print(f"Enhanced Factor {i+1}: Enhanced latent vector: {enhanced_latent}")
        
        # Get the factor profile for this enhanced latent vector
        enhanced_factor_profile = probabilistic_layer(enhanced_latent.reshape(1, -1)).numpy()
        enhanced_factor_profiles.append(enhanced_factor_profile[0])
    
    # so using the representative-sample method rather than the enhanced method
    factor_profiles = np.array(factor_profiles)
    
    # Save probabilistic factor profiles (these are the interpretable ones)
    factors_save_path = os.path.join(model_dir, "probabilistic_factors.npy")
    np.save(factors_save_path, factor_profiles)
    print(f"Probabilistic factor profiles saved to {factors_save_path}")
    
    # Print factor profile statistics
    print(f"\nProbabilistic Factor Profile Statistics:")
    print(f"  Shape: {factor_profiles.shape}")
    print(f"  Row sums (should be ~1.0): {np.sum(factor_profiles, axis=1)}")
    print(f"  Min value: {np.min(factor_profiles):.6f}")
    print(f"  Max value: {np.max(factor_profiles):.6f}")
    print(f"  Mean value: {np.mean(factor_profiles):.6f}")
    
    # Also save the raw logits weights with the old name for backward compatibility
    legacy_weights_save_path = os.path.join(model_dir, "linear_weights.npy")
    np.save(legacy_weights_save_path, W_logits)
    print(f"Legacy linear weights (logits) saved to {legacy_weights_save_path}")

if __name__ == "__main__":
    main()


