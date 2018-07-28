import tensorflow as tf
from models import AutoEncoder as AE
from ops.blocks import conv_block
from ops.losses import ssim_loss, lap1_loss
from ops.layers import dense, flatten, reshape


class AutoEncoder(AE):
    def __init__(self, input_shape,
                 latent_dim,
                 first_filters=16,
                 last_activation='tanh',
                 normalization='batch',
                 downsampling='stride',
                 upsampling='deconv',
                 distances='mse',
                 is_dropout=False):
        super().__init__(input_shape,
                         latent_dim,
                         first_filters,
                         last_activation,
                         normalization,
                         downsampling,
                         upsampling,
                         distances,
                         is_dropout)

    def encode(self, x,
               reuse=False,
               is_training=True):
        with tf.variable_scope(self.name):
            with tf.variable_scope('Encoder') as vs:
                if reuse:
                    vs.reuse_variables()

                for i in range(3):
                    filters = self.first_filters * (2**i)
                    x = conv_block(x,
                                   filters=filters, sampling='same',
                                   is_training=is_training,
                                   **self.conv_block_params)
                    x = conv_block(x,
                                   filters=filters, sampling='down',
                                   is_training=is_training,
                                   **self.conv_block_params)
                self.feature_shape = x.get_shape().as_list()[1:]

                x = flatten(x)
                x = dense(x, 128, activation_='lrelu')
                x = dense(x, self.latent_dim)
            return x

    def decode(self, x,
               reuse=False,
               is_training=True):
        with tf.variable_scope(self.name):
            with tf.variable_scope('Decoder') as vs:
                if reuse:
                    vs.reuse_variables()
                x = dense(x, 128, activation_='lrelu')
                x = dense(x,
                          self.feature_shape[0]*self.feature_shape[1]*self.feature_shape[2],
                          activation_='lrelu')
                x = reshape(x, self.feature_shape)

                for i in range(2)[::-1]:
                    filters = self.first_filters * (2**i)
                    x = conv_block(x,
                                   filters=filters, sampling=self.upsampling,
                                   is_training=is_training,
                                   **self.conv_block_params)
                    x = conv_block(x,
                                   filters=filters, sampling='same',
                                   is_training=is_training,
                                   **self.conv_block_params)

                x = conv_block(x,
                               filters=self.first_filters, sampling=self.upsampling,
                               **self.conv_block_params)
                x = conv_block(x,
                               filters=self.channel,
                               sampling='same',
                               **self.last_conv_block_params)
            return x
