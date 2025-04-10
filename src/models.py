"""
Module: models
This module defines the encoder, dual decoder, and autoencoder models along with
custom loss layers using TensorFlow and Keras. It also provides an object oriented
wrapper class (AutoencoderModel) for easier model management.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, constraints, regularizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from src import config


# ============================================================================
# Custom Loss / Helper Layers
# ============================================================================

class IdentityLayer(layers.Layer):
    """
    A simple layer that returns its input. Useful for naming outputs.
    """
    def call(self, inputs):
        return tf.identity(inputs)


class ConsistencyLossLayer(layers.Layer):
    """
    Computes the cosine similarity loss between deep_output and linear_output,
    then scales the loss by a weight and adds it via self.add_loss.
    """
    def __init__(self, weight, **kwargs):
        super(ConsistencyLossLayer, self).__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        deep_out, linear_out = inputs
        # Normalize the vectors
        y_true_norm = tf.nn.l2_normalize(deep_out, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(linear_out, axis=-1)
        cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        loss_val = 1 - cosine_sim
        loss_val = tf.reduce_mean(loss_val)
        self.add_loss(self.weight * loss_val)
        return inputs


class CorrelationLossLayer(layers.Layer):
    """
    Computes the correlation loss between the input (squeezed) and the deep branch output.
    This loss encourages the deep branch to maintain similar correlation structure as the input.
    """
    def __init__(self, weight, **kwargs):
        super(CorrelationLossLayer, self).__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        model_input, deep_out = inputs
        # Squeeze the last dimension: (batch, 43, 1) -> (batch, 43)
        x_true = tf.squeeze(model_input, axis=-1)
        mean_true = tf.reduce_mean(x_true, axis=0, keepdims=True)
        mean_pred = tf.reduce_mean(deep_out, axis=0, keepdims=True)
        x_true_centered = x_true - mean_true
        x_pred_centered = deep_out - mean_pred
        batch_size = tf.cast(tf.shape(x_true)[0], tf.float32)
        cov_true = tf.matmul(tf.transpose(x_true_centered), x_true_centered) / (batch_size - 1)
        cov_pred = tf.matmul(tf.transpose(x_pred_centered), x_pred_centered) / (batch_size - 1)
        std_true = tf.sqrt(tf.linalg.diag_part(cov_true))
        std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
        corr_true = cov_true / (std_true[:, None] * std_true[None, :] + 1e-8)
        corr_pred = cov_pred / (std_pred[:, None] * std_pred[None, :] + 1e-8)
        loss_val = tf.reduce_mean(tf.square(corr_true - corr_pred))
        self.add_loss(self.weight * loss_val)
        return deep_out

# ============================================================================
# Model Building Functions
# ============================================================================

def build_encoder(n_clusters, input_shape=(43, 1)):
    """
    Build the encoder model with 5 convolutional blocks.
    Residual connections and pooling are applied in the first 4 blocks.
    The final dense layers project the features to a latent space of dimension n_clusters.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Block 1 ---
    conv1 = layers.Conv1D(32, 3, padding='same')(inputs)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    shortcut1 = layers.Conv1D(32, 1, padding='same')(inputs)
    skip1 = layers.Add()([conv1, shortcut1])
    skip1 = layers.LeakyReLU(alpha=0.2)(skip1)
    pool1 = layers.MaxPooling1D(2)(skip1)  # shape: (floor(43/2)=21, 32)

    # --- Block 2 ---
    conv2 = layers.Conv1D(64, 3, padding='same')(pool1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
    shortcut2 = layers.Conv1D(64, 1, padding='same')(pool1)
    skip2 = layers.Add()([conv2, shortcut2])
    skip2 = layers.LeakyReLU(alpha=0.2)(skip2)
    pool2 = layers.MaxPooling1D(2)(skip2)  # shape: (floor(21/2)=10, 64)

    # --- Block 3 ---
    conv3 = layers.Conv1D(128, 3, padding='same')(pool2)
    conv3 = layers.LeakyReLU(alpha=0.2)(conv3)
    shortcut3 = layers.Conv1D(128, 1, padding='same')(pool2)
    skip3 = layers.Add()([conv3, shortcut3])
    skip3 = layers.LeakyReLU(alpha=0.2)(skip3)
    pool3 = layers.MaxPooling1D(2)(skip3)  # shape: (floor(10/2)=5, 128)

    # --- Block 4 ---
    conv4 = layers.Conv1D(256, 3, padding='same')(pool3)
    conv4 = layers.LeakyReLU(alpha=0.2)(conv4)
    shortcut4 = layers.Conv1D(256, 1, padding='same')(pool3)
    skip4 = layers.Add()([conv4, shortcut4])
    skip4 = layers.LeakyReLU(alpha=0.2)(skip4)
    pool4 = layers.MaxPooling1D(2)(skip4)  # shape: (floor(5/2)=2, 256)

    # --- Block 5 (No pooling) ---
    conv5 = layers.Conv1D(512, 3, padding='same')(pool4)
    conv5 = layers.LeakyReLU(alpha=0.2)(conv5)
    shortcut5 = layers.Conv1D(512, 1, padding='same')(pool4)
    skip5 = layers.Add()([conv5, shortcut5])
    skip5 = layers.LeakyReLU(alpha=0.2)(skip5)  # shape: (2, 512)

    # Flatten and Dense layers
    flat = layers.Flatten()(skip5)  # 2*512 = 1024
    dense1 = layers.Dense(256)(flat)
    dense1 = layers.LeakyReLU(alpha=0.2)(dense1)

    # Latent vector: dimension equals n_clusters with linear activation for interpretability
    latent = layers.Dense(n_clusters, activation='linear', name='latent')(dense1)

    # Create encoder model and attach skip connections for use in the decoder
    model = models.Model(inputs, latent, name="Encoder")
    model.skips = [skip1, skip2, skip3, skip4]  # Exclude block 5 (bottleneck)
    return model


def build_dual_decoder(n_clusters, input_dim=43, encoder_skips=None, linear_l1=1e-5, linear_l2=1e-3):
    """
    Build the dual-branch decoder.
      - Deep Branch (Reconstruction): Uses a series of upsampling blocks and skip connections
        to reconstruct the input spectral data.
      - Linear Branch (Interpretability): A single Dense layer with a non-negativity constraint.
    """
    latent_inputs = layers.Input(shape=(n_clusters,))

    ### Deep Branch ###
    # Project latent vector to a feature map (2,512)
    x = layers.Dense(2 * 512)(latent_inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((2, 512))(x)

    # Upsampling Block 1: (2 -> 4)
    x = layers.UpSampling1D(size=2)(x)
    if encoder_skips is not None:
        # Use skip connection from encoder block 4 (shape: (2,256))
        skip4 = encoder_skips[3]
        skip4_up = layers.UpSampling1D(size=2)(skip4)  # shape: (4,256)
        x = layers.Concatenate()([x, skip4_up])
    x = layers.Conv1D(512, 3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsampling Block 2: (4 -> 8)
    x = layers.UpSampling1D(size=2)(x)
    if encoder_skips is not None:
        skip3 = encoder_skips[2]
        skip3_up = layers.UpSampling1D(size=2)(skip3)  # from ~5 to ~10
        skip3_up = layers.Cropping1D(cropping=(1, 1))(skip3_up)  # crop to (8,128)
        x = layers.Concatenate()([x, skip3_up])
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsampling Block 3: (8 -> 16)
    x = layers.UpSampling1D(size=2)(x)
    if encoder_skips is not None:
        skip2 = encoder_skips[1]
        skip2_up = layers.UpSampling1D(size=2)(skip2)  # (from 10 to 20)
        skip2_up = layers.Cropping1D(cropping=(2, 2))(skip2_up)  # crop to (16,64)
        x = layers.Concatenate()([x, skip2_up])
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsampling Block 4: (16 -> 32)
    x = layers.UpSampling1D(size=2)(x)
    if encoder_skips is not None:
        skip1 = encoder_skips[0]
        skip1_up = layers.UpSampling1D(size=2)(skip1)  # (from 21 to 42)
        skip1_up = layers.Cropping1D(cropping=(5, 5))(skip1_up)  # crop to (32,32)
        x = layers.Concatenate()([x, skip1_up])
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsampling Block 5: (32 -> ~64)
    x = layers.UpSampling1D(size=2)(x)
    # Crop the temporal dimension to obtain output length 43
    x = layers.Cropping1D(cropping=(10, 11))(x)
    x = layers.Conv1D(1, 3, padding='same', activation='linear')(x)
    deep_output = layers.Flatten()(x)  # Expected shape: (43,)

    ### Linear Branch (Interpretability) ###
    linear_output = layers.Dense(
        input_dim,
        activation='linear',
        kernel_constraint=constraints.NonNeg(),
        kernel_regularizer=regularizers.l1_l2(l1=linear_l1, l2=linear_l2),
        name="linear_output"
    )(latent_inputs)

    return models.Model(latent_inputs, [deep_output, linear_output], name="DualDecoder")


def build_autoencoder(encoder, dual_decoder, lambda1=0.5, lambda2=0.6):
    """
    Build the full autoencoder that combines the encoder and dual decoder.
    Applies additional custom loss layers to align the deep and linear branches.
    """
    inputs = layers.Input(shape=(43, 1), name="AE_Input")
    latent = encoder(inputs)
    deep_output, linear_output = dual_decoder(latent)

    # Name the outputs for clarity with IdentityLayer wrappers
    deep_out_named = IdentityLayer(name="deep_output")(deep_output)
    linear_out_named = IdentityLayer(name="linear_output")(linear_output)

    # Add consistency loss to align the deep and linear branch outputs
    consistency_outputs = ConsistencyLossLayer(lambda1)([deep_out_named, linear_out_named])
    # Add correlation loss: compare the original input with the deep branch output
    deep_final = CorrelationLossLayer(lambda2)([inputs, consistency_outputs[0]])

    # For supervised training, the model outputs both deep and linear branches in a dictionary.
    return models.Model(inputs, {"deep_output": deep_final, "linear_output": consistency_outputs[1]}, name="Autoencoder")


# ============================================================================
# Object-Oriented Wrapper for the Full Autoencoder
# ============================================================================

class AutoencoderModel:
    """
    A wrapper class that encapsulates the encoder, dual decoder, and autoencoder.
    Provides methods to compile, train, and save the autoencoder model.
    """
    def __init__(self, n_clusters=3, input_shape=(43, 1), lambda1=0.5, lambda2=0.6, input_dim=43, linear_l1=1e-5, linear_l2=1e-3):
        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.input_dim = input_dim
        self.linear_l1 = linear_l1
        self.linear_l2 = linear_l2
        
        self.encoder = build_encoder(n_clusters=self.n_clusters, input_shape=self.input_shape)
        self.dual_decoder = build_dual_decoder(n_clusters=self.n_clusters, input_dim=self.input_dim,
                                               encoder_skips=self.encoder.skips , linear_l1=self.linear_l1, linear_l2=self.linear_l2)
        self.autoencoder = build_autoencoder(self.encoder, self.dual_decoder, self.lambda1, self.lambda2)
    
    def compile(self, optimizer, loss, loss_weights):
        """
        Compile the autoencoder model.
        """
        self.autoencoder.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
    
    def summary(self):
        """
        Print the summary of the autoencoder.
        """
        self.autoencoder.summary()
    
    def fit(self, *args, **kwargs):
        """
        Train the autoencoder model.
        """
        return self.autoencoder.fit(*args, **kwargs)
    
    def save(self, filepath):
        """
        Save the autoencoder model to a file.
        """
        self.autoencoder.save(filepath)

    def load(cls, filepath):
        """
        Load the autoencoder model from a file. Assumes custom objects have been registered.
        Returns an instance of AutoencoderModel with the loaded autoencoder.
        """
        custom_objs = {
            'IdentityLayer': IdentityLayer,
            'ConsistencyLossLayer': ConsistencyLossLayer,
            'CorrelationLossLayer': CorrelationLossLayer,
            # If you add additional custom layers, include them here.
            'mse': MeanSquaredError()
        }
        loaded_model = load_model(filepath, custom_objects=custom_objs)
        # Create an instance with default hyperparameters and then replace the autoencoder with the loaded one.
        instance = cls()
        instance.autoencoder = loaded_model
        return instance
