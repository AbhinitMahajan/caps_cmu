
# Emission Source Profiling using Autoencoder-based Unsupervised Learning

![Project Banner](assets/img_method.png) 

## Project Title
**Interpretable Source Separation of ACSM Spectral Data using Deep Autoencoders**

## Overview
This project aims to separate and identify emission sources from Aerosol Chemical Speciation Monitor (ACSM) spectral data collected at the Lawrenceville site in Pittsburgh. It leverages deep convolutional autoencoders to generate interpretable factor profiles, similar to traditional Positive Matrix Factorization (PMF) methods.

Developed in affiliation with the **Center for Atmospheric Particle Studies (CAPS)** at **Carnegie Mellon University**, this tool is tailored for environmental researchers interested in atmospheric source apportionment.

---

## Problem & Motivation
Traditional source separation techniques like Non‑negative Matrix Factorization (NMF) and PMF are widely used for ACSM data but are limited in capturing complex nonlinear dependencies in the spectra. This project offers a deep learning–based alternative that:

- Reconstructs spectral data with high fidelity.  
- Produces **interpretable and non‑negative linear factor outputs**.  
- Incorporates tailored loss functions to maintain structural consistency and correlation patterns.

---

## ⚙️ Methodology
The autoencoder architecture includes:

1. **Encoder:**  
   4 convolutional blocks (32→64→128→256 filters) with residual connections and max pooling, followed by a 512-filter bottleneck, compressing input to a K-dimensional latent space.

2. **Dual‑Branch Decoder:**  
   - **Deep Branch:** Reconstructs spectrum via upsampling with skip connections from encoder.  
   - **Probabilistic Branch:** Produces PMF-style factor profiles via temperature-scaled softmax over factor logits.

### Loss Functions
- **MSE Loss:** Supervises the deep branch for accurate spectrum reconstruction.  
- **PMF KL Loss:** Enforces probabilistic reconstruction where each sample is a mixture of global factor profiles.  
- **Consistency Loss:** Aligns deep and linear reconstructions via cosine similarity.  
- **Correlation Loss:** Preserves feature-wise correlation structure of the input spectra.  
- **Orthogonality Penalty:** Encourages diverse, non-overlapping factor profiles.  
- **Entropy Sparsity:** Promotes sharp per-sample factor assignments for clearer source attribution.

---

## Usage

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
Python 3.8+ recommended.

### Step 2: Train the Model
```bash
python src/training.py
```
You will be prompted for:
- Raw data file containing the ACSM data (in `data/raw/`)
- Number of epochs, batch size, number of clusters or factors
- Learning rate, consistency loss weight (`lambda1`), correlation loss weight (`lambda2`)
- Linear branch regularization (`l1`, `l2`)
- **Temperature:** Controls factor profile sharpness (lower = more focused, higher = smoother). Default: 1.0
- **Orthogonality Weight:** Promotes diverse factor profiles (higher = more distinct factors). Default: 0.01
- **Entropy Weight:** Encourages sparse factor usage per sample (higher = sharper assignments). Default: 0.001

Trained artifacts are saved to:
```
saved_models/
├── autoencoder_model.h5
├── factor_logits_weights.npy
└── probabilistic_factors.npy  # PMF-comparable factor profiles
```

### Step 3: Run Experiments
```bash
python run_experiments.py
```
Automated parameter sweep across factor counts, orthogonality weights, and entropy weights.

### Step 4: Analyze Results
```bash
python analyze_experiments.py    # Factor distinctiveness vs NMF
python calculate_q_metrics.py    # Q/Qexp fit quality metrics
```

### Step 5: Visualize Results
Open `test.ipynb` or use `src/visualisation.py` to:
- Load `probabilistic_factors.npy` (PMF-style factor profiles)
- Plot heatmaps and bar charts of factor profiles
- Compare with NMF profiles using correlation metrics

---

## Quick Start Guide

### Single Model Training (`src/training.py`)

**When to use:** Train a single model with custom parameters for exploration or quick testing.

**Steps:**

1. **Activate virtual environment** (if using one):
   ```bash
   source repro_test_env/bin/activate  # Linux/Mac
   # or
   repro_test_env\Scripts\activate     # Windows
   ```

2. **Run the training script**:
   ```bash
   python src/training.py
   ```

3. **Follow interactive prompts:**
   - **Data file:** Enter filename (default: `Spectra_Abhin_reduced.csv`) located in `data/raw/`
   - **Epochs:** Number of training iterations (default: 400)
   - **Batch size:** Samples per batch (default: 512)
   - **Number of clusters/factors:** Desired factor count (default: 3)
   - **Lambda1:** Consistency loss weight (default: 0.5)
   - **Lambda2:** Correlation loss weight (default: 0.6)
   - **Learning rate:** Optimizer step size (default: 1e-4)
   - **Linear L1/L2:** Regularization weights (defaults: 1e-5, 1e-3)
   - **Temperature:** Factor profile sharpness (default: 1.0)
   - **Orthogonality weight:** Factor diversity (default: 0.01)
   - **Entropy weight:** Assignment sparsity (default: 0.001)

4. **Output files saved to `saved_models/`:**
   - `autoencoder_model.h5` - Complete trained model
   - `factor_logits_weights.npy` - Raw factor logits (K × features)
   - `probabilistic_factors.npy` - Normalized factor profiles (K × features)
   - `linear_weights.npy` - Linear decoder weights

---

### Batch Experiment Runner (`run_experiments.py`)

**When to use:** Run multiple experiments with different factor counts and parameter combinations systematically.

**Steps:**

1. **Activate virtual environment** (if using one):
   ```bash
   source repro_test_env/bin/activate  # Linux/Mac
   # or
   repro_test_env\Scripts\activate     # Windows
   ```

2. **Run the experiment script**:
   ```bash
   python run_experiments.py
   ```

3. **Enter factor profiles** (comma-separated):
   ```
   Enter factor profiles (comma-separated, e.g., 3,4,5 or 5,6,7,8): 3,4,5,6
   ```
   - Script creates one experiment per factor count
   - Duplicates are removed and values are sorted

4. **Review default parameters:**
   - **Fixed:** temperature (0.1), entropy_weight (0.001), ortho_weight (1.0)
   - **Default:** epochs (300), batch_size (512), learning_rate (1e-4), etc.
   - All parameters are displayed with current values

5. **Modify parameters (optional):**
   - Answer `yes` when prompted to change parameters
   - Select parameter by number from the displayed list
   - Enter new value (type is preserved: int, float, or string)
   - Repeat or select "Done" when finished

6. **Confirm and run:**
   - Review final parameter summary
   - Answer `yes` to proceed with experiments
   - Script creates timestamped directory: `experiment_results_YYYYMMDD_HHMMSS/`

7. **Each experiment folder contains:**
   - `probabilistic_factors.npy` - Factor profiles (K × features)
   - `factor_contributions.npy` - Sample contributions (samples × K)
   - `factor_logits_weights.npy` - Raw logits before softmax
   - `autoencoder_model.h5` - Trained model
   - `training_history.npz` - Loss curves
   - `parameters.json` - Experiment configuration

**Example workflow:**
```bash
$ python run_experiments.py
Enter factor profiles: 4,5,6
# Confirm and run
# Results saved to experiment_results_20251209_114437/
```

---

### Analysis Scripts

After running experiments, use these scripts to evaluate and compare results:

#### Factor Analysis (`analyze_experiments.py`)

**Purpose:** Compare probabilistic factors with NMF baseline and assess factor distinctiveness.

**Usage:**
```bash
python analyze_experiments.py
```

**What it does:**
- Loads probabilistic factors from experiment directories
- Performs NMF decomposition for comparison
- Computes cross-correlations between autoencoder and NMF factors
- Calculates intra-method correlations (distinctiveness metric)
- Finds optimal factor matching using Hungarian algorithm

**Output saved to `analysis_results_*/`:**
- `correlation_matrix_*.png` - Cross-correlation heatmaps (autoencoder vs NMF)
- `matched_factors_*.png` - Side-by-side comparisons of matched factors
- `intra_correlation_trends.png` - Distinctiveness trends vs number of factors
- `analysis_summary.csv` - Quantitative metrics (correlations, distinctiveness)

**Key insights:**
- How distinct factors are within each method (lower intra-correlation = more distinct)
- How well autoencoder factors match NMF factors (cross-correlation)
- Optimal number of factors based on distinctiveness

---

#### Q/Qexp Metrics (`calculate_q_metrics.py`)

**Purpose:** Evaluate model fit quality using Q/Qexp ratio (standard metric in source apportionment).

**Usage:**
```bash
python calculate_q_metrics.py
```

**What it does:**
- Loads factor contributions and profiles from experiment directories
- Reconstructs spectra using saved contributions (no approximation)
- Calculates Q (weighted sum of squared residuals)
- Computes Qexp (expected Q based on degrees of freedom)
- Compares Q/Qexp ratios across different epoch counts and factor numbers

**Output saved to `analysis_results_q_metrics/`:**
- `q_qexp_direct_sqrt.png` - 4-panel comprehensive view (Q/Qexp, Q, log Q, MSE)
- `q_comparison_focused_sqrt.png` - Focused Q and Q/Qexp plots
- `q_qexp_direct_sqrt.csv` - All metrics in tabular format

**Key insights:**
- Q/Qexp ratio close to 1.0 indicates good fit
- Optimal number of factors (elbow point in Q/Qexp curve)
- Effect of training epochs on model performance
- Comparison across different hyperparameter settings

**Note:** Requires `factor_contributions.npy` files in experiment directories (automatically saved by `run_experiments.py`).

---

## Project Structure
```
caps_cmu/
├── data/
│   ├── raw/                # Raw ACSM CSV files
│   └── processed/         # Normalized data outputs
├── saved_models/           # Trained models and factor profiles
├── experiment_results_*/   # Timestamped experiment outputs
├── analysis_results_*/     # Analysis outputs (orthogonality, entropy, q_metrics)
├── src/
│   ├── config.py           # Seed settings & paths
│   ├── data_preprocessing.py
│   ├── models.py           # Encoder, decoder, autoencoder classes
│   ├── training.py         # CLI for interactive training
│   └── visualisation.py    # Plotting utilities
├── run_experiments.py      # Automated parameter sweep experiments
├── analyze_experiments.py  # Factor analysis vs NMF comparison
├── calculate_q_metrics.py  # Q/Qexp metric calculation
├── test.ipynb              # Notebook for evaluation & comparison
├── requirements.txt
└── README.md
```

---

## Contact
For questions or collaborations, reach out to Abhinit Mahajan at  
✉️ abhinitmahajan@cmu.edu

---

## 📄 License
This project is licensed under the MIT License. See LICENSE for details.

---

## Acknowledgements
Developed in affiliation and supervision with   
**Prof. Albert Prestro**
**Center for Atmospheric Particle Studies (CAPS)**  
**Carnegie Mellon University**
