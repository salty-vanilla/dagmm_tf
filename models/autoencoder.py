import tensorflow as tf
from ops.losses import ssim_loss, lap1_loss
import numpy as np


class AutoEncoder:
    def __init__(self, input_shape,
                 latent_dim,
                 first_filters=16,
                 last_activation='tanh',
                 normalization='batch',
                 downsampling='stride',
                 upsampling='deconv',
                 distances='mse',
                 is_dropout=False):
        self.input_shape = input_shape
        self.channel = input_shape[-1]
        self.latent_dim = latent_dim
        self.first_filters = first_filters
        self.last_activation = last_activation
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.distances = distances
        self.is_dropout = is_dropout
        self.feature_shape = None
        self.name = 'model/autoencoder'

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'relu',
                                  'normalization': self.normalization}

        self.last_conv_block_params = {'kernel_initializer': 'he_normal',
                                       'activation_': self.last_activation,
                                       'normalization': None}

    def __call__(self, x,
                 reuse=False,
                 is_training=True):
        return self.decode(self.encode(x,
                                       reuse=reuse,
                                       is_training=is_training),
                           reuse=reuse,
                           is_training=is_training)

    def encode(self, x,
               reuse=False,
               is_training=True):
        raise NotImplementedError

    def decode(self, x,
               reuse=False,
               is_training=True):
        raise NotImplementedError

    def calc_distance(self, y_true,
                      y_pred,
                      reuse=False,
                      is_training=True):
        distances = [self.distances] if isinstance(self.distances, str) \
                    else self.distances
        z_d = []
        for d in distances:
            if d == 'mse':
                _z_d = tf.reduce_mean((y_true - y_pred)**2, axis=[1, 2, 3])
            elif d == 'ssim':
                _z_d = ssim_loss(y_true, y_pred)
            elif d == 'lap1':
                max_leval = int(np.log2(np.minimum(*self.input_shape[:2])))
                _z_d = lap1_loss(y_true, y_pred, max_level=max_leval)
            else:
                raise ValueError
            z_d.append(tf.expand_dims(_z_d, axis=-1))

        z_d = z_d[0] if len(z_d) == 0 else tf.concat(z_d, axis=-1)
        return z_d

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def encoder_vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name and 'Encoder' in var.name]

    @property
    def update_ops(self):
        return [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in ops.name]

    @property
    def encoder_update_ops(self):
        return [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if self.name in ops.name and 'Encoder' in ops.name]
