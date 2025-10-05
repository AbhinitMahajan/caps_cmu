# Probabilistic Factor Profiles Implementation

## Overview
Successfully implemented **Option 1: Probabilistic Factor Profiles** to replace raw linear weights with interpretable, PMF-compatible factor profiles. This addresses all 5 identified problems with the original linear branch weights.

## ✅ Problems Solved

### 1. **Raw weights have no bounds** → **FIXED**
- **Before**: Raw weights could be any positive value (0.001 to 10.0+)
- **After**: Softmax normalization ensures all values in [0,1] range

### 2. **No normalization** → **FIXED** 
- **Before**: Factor profiles had arbitrary sums
- **After**: Each factor profile sums to exactly 1.0 (perfect PMF compatibility)

### 3. **Scale varies dramatically** → **FIXED**
- **Before**: Some factors dominated due to scale differences
- **After**: All factors use same probability scale (0-1)

### 4. **Hard to interpret magnitude** → **FIXED**
- **Before**: Weight of 0.05 vs 0.5 had unclear meaning
- **After**: 0.05 = 5% contribution, 0.5 = 50% contribution (clear interpretation)

### 5. **No probabilistic interpretation** → **FIXED**
- **Before**: Raw linear combinations with no clear meaning
- **After**: Direct probability distributions over chemical species

## 🔧 Implementation Details

### New Components Added

#### 1. **ProbabilisticFactorLayer Class** (`src/models.py`)
```python
class ProbabilisticFactorLayer(layers.Layer):
    def __init__(self, n_features, temperature=1.0, **kwargs):
        # Temperature controls factor sharpness
        # Lower temperature = sharper, more focused factors
        # Higher temperature = smoother, more diffuse factors
```

**Features:**
- Softmax normalization for probability distributions
- Temperature control for factor sharpness
- Built-in interpretability metrics (concentration, entropy)
- Automatic model metrics tracking

#### 2. **Enhanced Training Script** (`src/training.py`)
- Added temperature parameter for factor sharpness control
- Saves both raw logits and probabilistic factor profiles
- Provides detailed factor statistics during training
- Backward compatibility with legacy weight format

#### 3. **Advanced Visualization** (`src/visualisation.py`)
- New `load_and_plot_probabilistic_factors()` function
- Interpretability statistics (concentration, dominant species)
- Enhanced bar charts with sum annotations
- Improved heatmap formatting

#### 4. **Updated Notebook** (`test.ipynb`)
- Automatic detection of probabilistic vs legacy factors
- Comprehensive factor analysis with statistics
- Enhanced comparison capabilities with NMF

## 📊 New Output Files

After training, you'll get these new files in `saved_models/`:

1. **`probabilistic_factors.npy`** - The interpretable factor profiles (each row sums to 1.0)
2. **`factor_logits_weights.npy`** - Raw logits weights (for model reconstruction)
3. **`linear_weights.npy`** - Legacy format (backward compatibility)

## 🎯 Key Benefits

### **Scientific Interpretability**
- **Direct PMF comparison**: Factor profiles sum to 1.0 just like PMF
- **Clear species contributions**: "Factor 1 assigns 15% to m/z 44"
- **Concentration metrics**: Track how focused each factor is
- **Dominant species identification**: Automatically identify key m/z ratios

### **Enhanced Analysis**
- **Temperature control**: Adjust factor sharpness (0.5 = very focused, 2.0 = diffuse)
- **Built-in metrics**: Concentration, entropy, diversity automatically tracked
- **Comprehensive statistics**: Detailed factor analysis in training output
- **Visual enhancements**: Better plots with interpretability annotations

### **Backward Compatibility**
- **Legacy support**: Old models still work with existing code
- **Gradual migration**: Can compare old vs new approaches
- **File compatibility**: Maintains existing file structure

## 🚀 Usage Instructions

### **Training with Probabilistic Factors**
```bash
python src/training.py
# When prompted for temperature, use:
# - 0.5-0.8: Very focused factors (recommended for clear source separation)
# - 1.0: Balanced factors (default)
# - 1.5-2.0: More diffuse factors (if you want broader source profiles)
```

### **Analyzing Results**
```python
# In Jupyter notebook or Python script
from src.visualisation import load_and_plot_probabilistic_factors

load_and_plot_probabilistic_factors(
    factors_path='saved_models/probabilistic_factors.npy',
    data_csv='data/raw/Spectra_Abhin_reduced.csv',
    plot_statistics=True  # Shows detailed interpretability metrics
)
```

## 📈 Expected Improvements

### **Factor Quality**
- **Sharper source separation**: Temperature control allows fine-tuning
- **Better PMF alignment**: Direct probability interpretation
- **Clearer chemical meaning**: Each value represents actual contribution percentage

### **Scientific Communication**
- **Atmospheric scientist friendly**: Values directly interpretable
- **Publication ready**: Clear, standardized factor profiles
- **Reproducible results**: Deterministic with proper seeding

### **Analysis Capabilities**
- **Enhanced comparison**: Direct comparison with PMF results
- **Source identification**: Clear dominant species per factor
- **Quality metrics**: Built-in concentration and diversity measures

## 🔬 Scientific Impact

This implementation transforms your deep learning approach from a "black box" into a **transparent, interpretable tool** that atmospheric scientists can:

1. **Understand**: Each factor profile is a clear probability distribution
2. **Compare**: Direct compatibility with PMF methodology
3. **Validate**: Built-in quality metrics and statistics
4. **Publish**: Standardized, interpretable results

Your autoencoder now provides the **best of both worlds**: the power of deep learning to capture nonlinear relationships, combined with the interpretability and scientific rigor that makes PMF valuable to the atmospheric science community.

## 🎉 Ready to Use!

The implementation is complete and ready for training. Simply run:
```bash
python src/training.py
```

And you'll get interpretable, PMF-compatible factor profiles that solve all the original problems while maintaining the advanced capabilities of your deep learning approach.
