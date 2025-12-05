# Experiments and Analysis Results

## Experiment Results Directories

### `experiment_results_20251111_164144` (50 epochs)
- **Purpose**: Training with 50 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-9
- **Status**: Used in Q metrics analysis

### `experiment_results_20251118_034917` (60 epochs)
- **Purpose**: Training with 60 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-9
- **Status**: Used in Q metrics analysis

### `experiment_results_20251118_032307` (100 epochs)
- **Purpose**: Training with 100 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-9
- **Status**: Used in Q metrics analysis

### `experiment_results_20251118_022907` (250 epochs)
- **Purpose**: Training with 250 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-9
- **Status**: Used in Q metrics analysis

### `experiment_results_20251103_164022` (600 epochs)
- **Purpose**: Training with 600 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-9
- **Status**: Used in Q metrics analysis

### `experiment_results_20251111_131950` (2000 epochs)
- **Purpose**: Training with 2000 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-6
- **Status**: Used in Q metrics analysis

### `experiment_results_20251118_040759` (25 epochs)
- **Purpose**: Training with 25 epochs
- **Parameters**: temp=0.1, ortho=1.0, entropy=0.001
- **Factors**: 3-9
- **Status**: **NOT USED** - Can be deleted

**Each directory contains:**
- `probabilistic_factors.npy` - Factor profiles
- `factor_logits_weights.npy` - Raw logits
- `autoencoder_model.h5` - Trained model
- `training_history.npz` - Loss history
- `parameters.json` - Experiment config

## Analysis Results Directories

### `analysis_results_orthogonality/`
- **Source**: `experiment_orthogonality/` directory
- **Study**: Orthogonality weight impact (100, 10, 1, 0.1, 0.01, 0.001)
- **Fixed**: entropy=0.001, temp=0.1
- **Factors**: 3-9
- **Contents**:
  - Correlation matrices (probabilistic vs NMF)
  - Matched factor plots (side-by-side comparisons)
  - Intra-correlation trends
  - Summary CSV

### `analysis_results_entropy/`
- **Source**: `experiment_entropy/` directory
- **Study**: Entropy weight impact (100, 10, 1, 0.1, 0.01, 0.001)
- **Fixed**: ortho=10, temp=0.1
- **Factors**: 3-9
- **Contents**:
  - Correlation matrices (probabilistic vs NMF)
  - Matched factor plots (side-by-side comparisons)
  - Intra-correlation trends
  - Summary CSV

### `analysis_results_q_metrics/`
- **Source**: Multiple `experiment_results_2025*` directories
- **Study**: Q/Qexp metrics across different training epochs
- **Epochs**: 50, 60, 100, 250, 600, 2000
- **Fixed**: temp=0.1, ortho=1.0, entropy=0.001
- **Contents**:
  - Q/Qexp ratio plots (4-panel comprehensive)
  - Focused comparison plots (Q and Q/Qexp)
  - Summary CSV with all metrics


