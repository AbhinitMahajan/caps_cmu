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


# ============================================================================
# Model Building Functions
# ============================================================================

def build_autoencoder(
    n_clusters,
    input_dim,
    lambda1=0.5,
    lambda2=0.6,
    linear_l1=1e-5,
    linear_l2=1e-3
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

    # Linear decoder branch
    lin_out = layers.Dense(
        input_dim,
        activation='linear',
        kernel_constraint=constraints.NonNeg(),
        kernel_regularizer=regularizers.l1_l2(l1=linear_l1, l2=linear_l2),
        name='linear_output'
    )(latent)

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
        linear_l2=1e-3
    ):
        input_dim = input_shape[0]
        self.model = build_autoencoder(
            n_clusters, input_dim, lambda1=lambda1, lambda2=lambda2,
            linear_l1=linear_l1, linear_l2=linear_l2
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
            'MeanSquaredError': MeanSquaredError
        }
        model = load_model(filepath, custom_objects=custom_objs)
        inst = cls(0, (model.input_shape[1], 1), 0, 0)
        inst.model = model
        return inst
