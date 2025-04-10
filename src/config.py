# src/config.py
import os
import random
import numpy as np
import tensorflow as tf

# Set a global seed for reproducibility
SEED_VALUE = 42
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Define global paths:
DATA_RAW_DIR = os.path.join(os.getcwd(), 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(os.getcwd(), 'data', 'processed')
