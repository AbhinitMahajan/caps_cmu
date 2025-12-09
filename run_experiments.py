#!/usr/bin/env python3
"""
Experiment runner for autoencoder training with multiple parameter combinations.
Runs experiments across different factor profiles, temperature, and orthogonality weights.
"""

import numpy as np
import os
import sys
import json
from datetime import datetime
import tensorflow as tf
import random

# Add project root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import required modules
from src import config
from src.data_preprocessing import load_and_preprocess_data, prepare_data
from src.models import AutoencoderModel
from tensorflow.keras.optimizers import Adam


def ensure_reproducibility(seed_value):
    """
    Apply all seed settings to ensure reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.experimental.enable_op_determinism()


def run_single_experiment(params, experiment_id, base_output_dir):
    """
    Run a single experiment with the given parameters.
    
    Args:
        params: Dictionary of hyperparameters
        experiment_id: Unique identifier for this experiment
        base_output_dir: Base directory for saving results
    
    Returns:
        Dictionary with experiment results and paths
    """
    print(f"\n{'='*80}")
    print(f"Starting Experiment {experiment_id}")
    print(f"{'='*80}")
    print(f"Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Create experiment-specific directory
    exp_name = f"factors{params['n_clusters']}_temp{params['temperature']}_ortho{params['ortho_weight']}_entropy{params['entropy_weight_a']}"
    exp_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment parameters
    params_path = os.path.join(exp_dir, "parameters.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to: {params_path}")
    
    # Ensure reproducibility
    ensure_reproducibility(config.SEED_VALUE)
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(params['file_name'])
    X_input, X_target = prepare_data(df)
    input_dim = X_input.shape[1]
    print(f"Input shape: {X_input.shape}")
    
    # Create train/validation split
    n_samples = X_input.shape[0]
    split_idx = int(0.75 * n_samples)
    
    X_train = X_input[:split_idx]
    X_val = X_input[split_idx:]
    y_train = X_target[:split_idx]
    y_val = X_target[split_idx:]
    
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    
    # Build model
    print("Building model...")
    physical_gpus = tf.config.list_physical_devices('GPU')
    device_scope = '/GPU:0' if physical_gpus else '/CPU:0'
    print(f"Using device: {device_scope}")
    
    with tf.device(device_scope):
        ae_model = AutoencoderModel(
            n_clusters=params['n_clusters'],
            input_shape=(input_dim, 1),
            lambda1=params['lambda1'],
            lambda2=params['lambda2'],
            linear_l1=params['linear_l1'],
            linear_l2=params['linear_l2'],
            temperature=params['temperature'],
            ortho_weight=params['ortho_weight'],
            entropy_weight_a=params['entropy_weight_a']
        )
    
    # Compile model
    optimizer = Adam(learning_rate=params['learning_rate'])
    ae_model.compile(
        optimizer=optimizer,
        loss={"deep_output": "mse", "linear_output": "mse"},
        loss_weights={"deep_output": 1.0, "linear_output": 0.0}
    )
    
    print("\nModel compiled successfully")
    
    # Train model
    print(f"\nTraining for {params['epochs']} epochs...")
    with tf.device(device_scope):
        history = ae_model.fit(
            X_train,
            {"deep_output": y_train, "linear_output": y_train},
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, {"deep_output": y_val, "linear_output": y_val}),
            shuffle=False,
            verbose=1
        )
    
    print("Training completed!")
    
    # Save model
    model_save_path = os.path.join(exp_dir, "autoencoder_model.h5")
    ae_model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Extract and save probabilistic factor profiles
    print("\nExtracting probabilistic factor profiles...")
    pmf_kl_layer = ae_model.model.get_layer('pmf_kl')
    probabilistic_layer = pmf_kl_layer.prob_layer
    factor_logits_layer = probabilistic_layer.factor_logits
    W_logits = factor_logits_layer.get_weights()[0]  # (K, F)
    
    # Save raw logits
    logits_save_path = os.path.join(exp_dir, "factor_logits_weights.npy")
    np.save(logits_save_path, W_logits)
    print(f"Factor logits saved to: {logits_save_path}")
    
    # Compute probabilistic factor profiles (softmax of logits)
    temperature_val = params['temperature']
    logits_scaled = W_logits / temperature_val
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled, axis=1, keepdims=True))
    factor_profiles = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Save probabilistic factors
    factors_save_path = os.path.join(exp_dir, "probabilistic_factors.npy")
    np.save(factors_save_path, factor_profiles)
    print(f"Probabilistic factors saved to: {factors_save_path}")
    
    # Print statistics
    print(f"\nFactor Profile Statistics:")
    print(f"  Shape: {factor_profiles.shape}")
    print(f"  Row sums: {np.sum(factor_profiles, axis=1)}")
    print(f"  Min: {np.min(factor_profiles):.6f}, Max: {np.max(factor_profiles):.6f}")
    print(f"  Mean: {np.mean(factor_profiles):.6f}")
    
    # Extract and save factor contributions (G matrix)
    print("\nExtracting factor contributions...")
    encoder_model = tf.keras.Model(
        inputs=ae_model.model.input,
        outputs=ae_model.model.get_layer('latent').output
    )
    
    # Get latent vectors for the full dataset
    latent_vectors = encoder_model.predict(X_input, batch_size=params['batch_size'], verbose=0)
    
    # Apply softmax to get probabilistic contributions (same as PMFKLLossLayer)
    contributions = tf.nn.softmax(latent_vectors, axis=-1).numpy()
    
    # Save contributions
    contributions_save_path = os.path.join(exp_dir, "factor_contributions.npy")
    np.save(contributions_save_path, contributions)
    print(f"Factor contributions saved to: {contributions_save_path}")
    
    # Print contribution statistics
    print(f"  Shape: {contributions.shape}")
    print(f"  Mean per factor: {np.mean(contributions, axis=0).round(4)}")
    
    # Verify reconstruction
    reconstruction = contributions @ factor_profiles
    mse_check = np.mean((X_target - reconstruction) ** 2)
    print(f"  Reconstruction MSE: {mse_check:.6f}")
    
    # Save training history
    history_path = os.path.join(exp_dir, "training_history.npz")
    np.savez(history_path, **history.history)
    print(f"Training history saved to: {history_path}")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    result = {
        'experiment_id': experiment_id,
        'experiment_name': exp_name,
        'directory': exp_dir,
        'final_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
    }
    
    print(f"\nExperiment {experiment_id} completed!")
    print(f"Final training loss: {result['final_loss']:.6f}")
    if result['final_val_loss']:
        print(f"Final validation loss: {result['final_val_loss']:.6f}")
    
    return result


def get_factor_profiles_input():
    """
    Get factor profiles list from user input.
    Returns a list of integers.
    """
    while True:
        user_input = input("\nEnter factor profiles (comma-separated, e.g., 3,4,5 or 5,6,7,8): ").strip()
        if not user_input:
            print("Error: Please provide at least one factor profile.")
            continue
        
        try:
            factors = [int(x.strip()) for x in user_input.split(',')]
            if len(factors) == 0:
                print("Error: Please provide at least one factor profile.")
                continue
            if any(f <= 0 for f in factors):
                print("Error: Factor profiles must be positive integers.")
                continue
            return sorted(set(factors))  # Remove duplicates and sort
        except ValueError:
            print("Error: Please enter valid integers separated by commas.")
            continue


def display_parameters(fixed_params, default_params):
    """
    Display current fixed and default parameters.
    """
    print("\n" + "="*80)
    print("CURRENT PARAMETERS")
    print("="*80)
    
    print("\nFixed Parameters:")
    for key, value in fixed_params.items():
        print(f"  {key}: {value}")
    
    print("\nDefault Parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    print("="*80)


def get_user_parameter_changes(fixed_params, default_params):
    """
    Interactive function to get parameter changes from user.
    Returns updated fixed_params and default_params dictionaries.
    """
    fixed_params_updated = fixed_params.copy()
    default_params_updated = default_params.copy()
    
    while True:
        change_input = input("\nDo you want to change any parameters? (yes/no): ").strip().lower()
        if change_input not in ['yes', 'y', 'no', 'n']:
            print("Please enter 'yes' or 'no'.")
            continue
        
        if change_input in ['no', 'n']:
            break
        
        # Show options
        print("\nWhich parameter would you like to change?")
        print("\nFixed Parameters:")
        fixed_keys = list(fixed_params_updated.keys())
        for i, key in enumerate(fixed_keys, 1):
            print(f"  {i}. {key} (current: {fixed_params_updated[key]})")
        
        print("\nDefault Parameters:")
        default_keys = list(default_params_updated.keys())
        for i, key in enumerate(default_keys, len(fixed_keys) + 1):
            print(f"  {i}. {key} (current: {default_params_updated[key]})")
        
        print(f"  {len(fixed_keys) + len(default_keys) + 1}. Done (no more changes)")
        
        try:
            choice = int(input("\nEnter option number: ").strip())
            
            if choice == len(fixed_keys) + len(default_keys) + 1:
                break
            
            if 1 <= choice <= len(fixed_keys):
                # Changing fixed parameter
                param_key = fixed_keys[choice - 1]
                current_value = fixed_params_updated[param_key]
                new_value_str = input(f"Enter new value for {param_key} (current: {current_value}): ").strip()
                
                # Try to convert to appropriate type
                if isinstance(current_value, float):
                    new_value = float(new_value_str)
                elif isinstance(current_value, int):
                    new_value = int(new_value_str)
                else:
                    new_value = new_value_str
                
                fixed_params_updated[param_key] = new_value
                print(f"✓ Updated {param_key}: {current_value} -> {new_value}")
                
            elif len(fixed_keys) + 1 <= choice <= len(fixed_keys) + len(default_keys):
                # Changing default parameter
                param_key = default_keys[choice - len(fixed_keys) - 1]
                current_value = default_params_updated[param_key]
                new_value_str = input(f"Enter new value for {param_key} (current: {current_value}): ").strip()
                
                # Try to convert to appropriate type
                if isinstance(current_value, float):
                    new_value = float(new_value_str)
                elif isinstance(current_value, int):
                    new_value = int(new_value_str)
                else:
                    new_value = new_value_str
                
                default_params_updated[param_key] = new_value
                print(f"✓ Updated {param_key}: {current_value} -> {new_value}")
            else:
                print("Invalid option. Please try again.")
                
        except ValueError:
            print("Error: Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")
    
    return fixed_params_updated, default_params_updated


def main():
    """
    Main function to run all experiments.
    """
    print("="*80)
    print("AUTOENCODER EXPERIMENT RUNNER")
    print("="*80)
    
    # Configure GPU
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
    
    # Get factor profiles from user
    factor_profiles_list = get_factor_profiles_input()
    print(f"\n✓ Factor profiles selected: {factor_profiles_list}")
    
    # Define default fixed parameters
    fixed_params = {
        'temperature': 0.1,
        'entropy_weight': 0.001,
        'ortho_weight': 1.0
    }
    
    # Define default parameters (kept constant)
    default_params = {
        'file_name': 'Spectra_Abhin_reduced.csv',
        'epochs': 300,
        'batch_size': 512,
        'lambda1': 0.5,
        'lambda2': 0.6,
        'learning_rate': 1e-4,
        'linear_l1': 1e-5,
        'linear_l2': 1e-3,
        'temperature': fixed_params['temperature'],
        'ortho_weight': fixed_params['ortho_weight'],
        'entropy_weight_a': fixed_params['entropy_weight']
    }
    
    # Display current parameters
    display_parameters(fixed_params, default_params)
    
    # Get user changes
    fixed_params, default_params = get_user_parameter_changes(fixed_params, default_params)
    
    # Update default_params with fixed_params values
    default_params['temperature'] = fixed_params['temperature']
    default_params['ortho_weight'] = fixed_params['ortho_weight']
    default_params['entropy_weight_a'] = fixed_params['entropy_weight']
    
    # Display final parameters
    print("\n" + "="*80)
    print("FINAL PARAMETERS")
    print("="*80)
    display_parameters(fixed_params, default_params)
    
    # Generate experiments (one for each number of factors)
    experiments = []
    for n_clusters in factor_profiles_list:
        exp_params = default_params.copy()
        exp_params['n_clusters'] = n_clusters
        experiments.append(exp_params)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total number of experiments: {len(experiments)}")
    print(f"  Factor profiles: {factor_profiles_list}")
    print(f"  Temperature: {fixed_params['temperature']}")
    print(f"  Entropy weight: {fixed_params['entropy_weight']}")
    print(f"  Orthogonality weight: {fixed_params['ortho_weight']}")
    
    # Confirm before proceeding
    confirm = input("\nProceed with experiments? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Experiments cancelled.")
        return
    
    print(f"\n{'='*80}\n")
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(ROOT, f"experiment_results_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"All results will be saved to: {base_output_dir}\n")
    
    # Run all experiments
    results = []
    for i, params in enumerate(experiments, 1):
        try:
            result = run_single_experiment(params, i, base_output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR in Experiment {i}:")
            print(f"{str(e)}")
            print(f"{'!'*80}\n")
            results.append({
                'experiment_id': i,
                'error': str(e)
            })
            continue
    
    # Save summary of all experiments
    summary_path = os.path.join(base_output_dir, "experiments_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"Results directory: {base_output_dir}")
    print(f"{'='*80}\n")
    
    # Print summary table
    print("\nExperiment Summary:")
    print(f"{'ID':<5} {'Factors':<8} {'Temp':<8} {'Ortho':<8} {'Entropy':<10} {'Final Loss':<15} {'Status'}")
    print("-" * 85)
    for result in results:
        exp_id = result['experiment_id']
        if 'error' in result:
            print(f"{exp_id:<5} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<15} FAILED")
        else:
            # Extract parameters from experiment name
            # Format: factors{N}_temp{T}_ortho{O}_entropy{E}
            name_parts = result['experiment_name'].split('_')
            factors = name_parts[0].replace('factors', '')
            temp = name_parts[1].replace('temp', '')
            ortho = name_parts[2].replace('ortho', '')
            entropy = name_parts[3].replace('entropy', '')
            loss = f"{result['final_loss']:.6f}"
            print(f"{exp_id:<5} {factors:<8} {temp:<8} {ortho:<8} {entropy:<10} {loss:<15} SUCCESS")
    
    print(f"\n{'='*80}")
    print(f"All results saved in: {base_output_dir}")
    print(f"Each experiment folder contains:")
    print(f"  - probabilistic_factors.npy (interpretable factor profiles)")
    print(f"  - factor_contributions.npy (factor contributions matrix)")
    print(f"  - factor_logits_weights.npy (raw logits)")
    print(f"  - autoencoder_model.h5 (trained model)")
    print(f"  - training_history.npz (loss history)")
    print(f"  - parameters.json (experiment parameters)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

