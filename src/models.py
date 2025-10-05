"""
Module: models
This module defines the encoder, dual decoder, and autoencoder models along with
custom loss layers using TensorFlow and Keras. It also provides an object oriented
wrapper class (AutoencoderModel) for easier model management.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, constraints, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import math
from src import config


# ============================================================================
# Custom Loss Layers
# ============================================================================

class IdentityLayer(layers.Layer):
    """
    Pass-through layer to name tensors.
    """
    def call(self, inputs):
        return tf.identity(inputs)


class ConsistencyLossLayer(layers.Layer):
    """
    Computes cosine-similarity loss between deep and linear branches.
    Adds weighted mean loss via self.add_loss and returns deep_out.
    """
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        deep_out, linear_out = inputs
        # 1 - cosine similarity per sample
        loss = 1 - tf.reduce_sum(
            tf.nn.l2_normalize(deep_out, axis=-1) * tf.nn.l2_normalize(linear_out, axis=-1),
            axis=-1
        )
        self.add_loss(self.weight * tf.reduce_mean(loss))
        return deep_out


class CorrelationLossLayer(layers.Layer):
    """
    Computes correlation-structure loss between model_input and deep branch output.
    Adds weighted mean squared difference of correlation matrices via self.add_loss.
    """
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        model_input, deep_out = inputs
        # squeeze to (batch, features)
        x_true = tf.squeeze(model_input, axis=-1)
        x_pred = tf.squeeze(deep_out, axis=-1)

        # center
        mean_true = tf.reduce_mean(x_true, axis=0, keepdims=True)
        mean_pred = tf.reduce_mean(x_pred, axis=0, keepdims=True)
        x_true_centered = x_true - mean_true
        x_pred_centered = x_pred - mean_pred

        # covariance
        batch_size = tf.cast(tf.shape(x_true)[0], tf.float32)
        cov_true = tf.matmul(x_true_centered, x_true_centered, transpose_a=True) / (batch_size - 1)
        cov_pred = tf.matmul(x_pred_centered, x_pred_centered, transpose_a=True) / (batch_size - 1)

        # correlation matrices
        std_true = tf.sqrt(tf.linalg.diag_part(cov_true))
        std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
        corr_true = cov_true / (std_true[:, None] * std_true[None, :] + 1e-8)
        corr_pred = cov_pred / (std_pred[:, None] * std_pred[None, :] + 1e-8)

        # mean squared difference
        loss_val = tf.reduce_mean(tf.square(corr_true - corr_pred))
        self.add_loss(self.weight * loss_val)
        return deep_out


class PMFKLLossLayer(layers.Layer):
    """
    Adds a PMF-style reconstruction KL loss:
    - Profiles P are the softmax over species of the rows of factor logits
    - Contributions a_i are the softmax over factors of the latent vector
    - Reconstruction y_hat_i = a_i @ P
    - Target y_i is the per-sample L1-normalized input spectrum

    Also adds an orthogonality penalty on P via off-diagonal energy of P @ P^T.
    """
    def __init__(self, prob_layer, ortho_weight=1e-3, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.prob_layer = prob_layer  # instance of ProbabilisticFactorLayer
        self.ortho_weight = ortho_weight
        self.eps = eps

    def call(self, inputs):
        model_input, latent = inputs  # model_input shape (batch, F, 1); latent (batch, K)

        # Normalize true spectra per sample (L1)
        x_true = tf.squeeze(model_input, axis=-1)  # (batch, F)
        x_sum = tf.reduce_sum(x_true, axis=1, keepdims=True)
        y_true = x_true / (x_sum + self.eps)

        # Global factor profiles from logits: P (K, F)
        logits = self.prob_layer.factor_logits.kernel  # (K, F)
        temperature = tf.convert_to_tensor(self.prob_layer.temperature, dtype=logits.dtype)
        P = tf.nn.softmax(logits / temperature, axis=-1)

        # Per-sample contributions from latent: a (batch, K)
        a = tf.nn.softmax(latent, axis=-1)

        # Reconstruction y_hat = a @ P
        y_hat = tf.matmul(a, P)

        # KL(y_true || y_hat)
        kl = tf.reduce_sum(y_true * (tf.math.log(y_true + self.eps) - tf.math.log(y_hat + self.eps)), axis=1)
        kl_loss = tf.reduce_mean(kl)

        # Orthogonality penalty on P
        gram = tf.matmul(P, P, transpose_b=True)  # (K, K)
        diag = tf.linalg.diag_part(gram)
        gram_off = gram - tf.linalg.diag(diag)
        ortho_loss = tf.reduce_mean(tf.square(gram_off))

        self.add_loss(kl_loss + self.ortho_weight * ortho_loss)
        return latent  # pass-through


class ProbabilisticFactorLayer(layers.Layer):
    """
    Creates interpretable probabilistic factor profiles using softmax normalization.
    Each factor profile sums to 1.0, providing clear probabilistic interpretation
    similar to PMF methodology.
    
    This layer transforms raw latent representations into probability distributions
    over chemical species (m/z features), making the factors interpretable and
    directly comparable to PMF results.
    """
    def __init__(self, n_features, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.temperature = temperature
        
    def build(self, input_shape):
        # Create the dense layer that maps from latent space to logits
        self.factor_logits = layers.Dense(
            self.n_features,
            activation='linear',
            name='factor_logits',
            kernel_initializer='glorot_uniform'
        )
        super().build(input_shape)
        
    def call(self, latent_vectors):
        # Get raw logits from latent space
        logits = self.factor_logits(latent_vectors)
        
        # Apply temperature-scaled softmax for controllable sharpness
        # Lower temperature = sharper distributions (more focused factors)
        # Higher temperature = smoother distributions (more diffuse factors)
        factor_profiles = tf.nn.softmax(logits / self.temperature, axis=-1)
        
        return factor_profiles
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
            'temperature': self.temperature
        })
        return config


# ============================================================================
# Model Building Functions
# ============================================================================

def build_autoencoder(
    n_clusters,
    input_dim,
    lambda1=0.5,
    lambda2=0.6,
    linear_l1=1e-5,
    linear_l2=1e-3,
    temperature=1.0
):
    inp = layers.Input(shape=(input_dim, 1), name='AE_Input')
    x = inp
    skips = []

    # Encoder: conv + residual + pool
    for filters in [32, 64, 128, 256]:
        y = layers.Conv1D(filters, 3, padding='same')(x)
        y = layers.LeakyReLU(negative_slope=0.2)(y)
        y = layers.Add()([
            y,
            layers.Conv1D(filters, 1, padding='same')(x)
        ])
        y = layers.LeakyReLU(negative_slope=0.2)(y)
        skips.append(y)
        x = layers.MaxPooling1D(2)(y)

    # Bottleneck
    y = layers.Conv1D(512, 3, padding='same')(x)
    y = layers.LeakyReLU(negative_slope=0.2)(y)
    y = layers.Add()([
        y,
        layers.Conv1D(512, 1, padding='same')(y)
    ])
    y = layers.LeakyReLU(negative_slope=0.2)(y)
    flat = layers.Flatten()(y)
    dense = layers.LeakyReLU(negative_slope=0.2)(layers.Dense(256)(flat))
    latent = layers.Dense(n_clusters, name='latent')(dense)

    # Dual decoder: deep reconstruction
    z = layers.LeakyReLU(negative_slope=0.2)(layers.Dense(2 * 512)(latent))
    z = layers.Reshape((2, 512))(z)
    decoder_filters = [512, 256, 128, 64]
    for i, filters in enumerate(decoder_filters):
        z = layers.UpSampling1D(2)(z)
        skip_tensor = skips[-(i + 1)]
        s = layers.UpSampling1D(2)(skip_tensor)

        # dynamic crop skip to match z
        def crop_fn(tensors):
            s_t, z_t = tensors
            diff = tf.shape(s_t)[1] - tf.shape(z_t)[1]
            start = diff // 2
            return s_t[:, start:start + tf.shape(z_t)[1], :]

        s = layers.Lambda(crop_fn, name=f'crop_skip_{i}')([s, z])
        z = layers.Concatenate()([z, s])
        z = layers.LeakyReLU(negative_slope=0.2)(layers.Conv1D(filters, 3, padding='same')(z))

    # Dynamic extra upsampling to next power of two >= input_dim
    length_loop = 2 ** (len(decoder_filters) + 1)
    overshoot = 1 << (input_dim - 1).bit_length()
    extra_ups = int(math.log2(overshoot / length_loop))
    for _ in range(extra_ups):
        z = layers.UpSampling1D(2)(z)

    # Final crop back to original length
    z = layers.Lambda(lambda t: t[:, :input_dim, :], name='final_crop')(z)
    deep_conv = layers.Conv1D(1, 3, padding='same', name='conv_deep')(z)
    deep_out = layers.Flatten(name='deep_flat')(deep_conv)

    # Probabilistic factor decoder branch
    # This replaces the raw linear branch with interpretable probabilistic factors
    prob_layer = ProbabilisticFactorLayer(
        n_features=input_dim,
        temperature=temperature,
        name='probabilistic_factors'
    )
    lin_out = prob_layer(latent)

    # PMF-style KL reconstruction loss and orthogonality regularization
    _ = PMFKLLossLayer(prob_layer=prob_layer, ortho_weight=1e-3, name='pmf_kl')([inp, latent])

    # Attach loss layers
    d_named = IdentityLayer(name='deep_named')(deep_out)
    l_named = IdentityLayer(name='lin_named')(lin_out)

    d_cons = ConsistencyLossLayer(lambda1, name='consistency')([d_named, l_named])
    d_exp = layers.Lambda(lambda x: tf.expand_dims(x, -1), name='expand_deep')(d_cons)
    d_final = CorrelationLossLayer(lambda2, name='correlation')([inp, d_exp])

    return models.Model(
        inputs=inp,
        outputs={'deep_output': d_final, 'linear_output': l_named},
        name='Autoencoder'
    )


class AutoencoderModel:
    """
    Wrapper for easier model management.
    """
    def __init__(
        self,
        n_clusters,
        input_shape,
        lambda1,
        lambda2,
        linear_l1=1e-5,
        linear_l2=1e-3,
        temperature=1.0
    ):
        input_dim = input_shape[0]
        self.model = build_autoencoder(
            n_clusters, input_dim, lambda1=lambda1, lambda2=lambda2,
            linear_l1=linear_l1, linear_l2=linear_l2, temperature=temperature
        )

    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def summary(self):
        return self.model.summary()

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        custom_objs = {
            'ConsistencyLossLayer': ConsistencyLossLayer,
            'CorrelationLossLayer': CorrelationLossLayer,
            'PMFKLLossLayer': PMFKLLossLayer,
            'ProbabilisticFactorLayer': ProbabilisticFactorLayer,
            'MeanSquaredError': MeanSquaredError
        }
        model = load_model(filepath, custom_objects=custom_objs)
        inst = cls(0, (model.input_shape[1], 1), 0, 0)
        inst.model = model
        return inst
