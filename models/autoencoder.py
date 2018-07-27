import tensorflow as tf


class AutoEncoder(tf.keras.Model):
    def __init__(self, input_shape,
                 latent_dim,
                 first_filters=16,
                 last_activation='tanh',
                 normalization='batch',
                 downsampling='stride',
                 upsampling='deconv',
                 is_dropout=False,
                 is_training=True):
        self._input_shape = input_shape
        self.latent_dim = latent_dim
        self.first_filters = first_filters
        self.last_activation = last_activation
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.is_dropout = is_dropout
        self.is_training = is_training

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'relu',
                                  'normalization': self.normalization,
                                  'is_training': self.is_training}

        self.last_conv_block_params = {'kernel_initializer': 'he_normal',
                                       'activation_': self.last_activation,
                                       'normalization': None,
                                       'is_training': self.is_training}

        super().__init__(name='model/autoencoder')

    def call(self, x,
             reuse=False,
             **kwargs):
        return self.decode(self.encode(x, reuse=reuse), reuse=reuse)

    def encode(self, x,
               reuse=False):
        raise NotImplementedError

    def decode(self, x,
               reuse=False):
        raise NotImplementedError

    def calc_distance(self, y_true,
                      y_pred,
                      reuse=False):
        raise NotImplementedError

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
