import tensorflow as tf
from ops.layers import dense, batch_norm, layer_norm, dropout, activation


class EstimationNetwork(tf.keras.Model):
    def __init__(self, input_shape,
                 dense_units,
                 normalization='batch',
                 is_dropout=False,
                 is_training=True,):
        self._input_shape = input_shape
        self.dense_units = dense_units
        self.normalization = normalization
        self.is_dropout = is_dropout
        self.is_training = is_training

        self.dense_params = {'kernel_initializer': 'he_normal',
                             'activation_': 'relu',
                             'normalization': self.normalization}

        super().__init__(name='model/estimation_network')

    def call(self, x,
             reuse=False,
             **kwargs):
        for u in self.dense_units[:-1]:
            x = dense(x, u)
            x = activation(x, self.dense_params['activation_'])
            if self.dense_params['normalization'] is not None:
                if self.dense_params['normalization'] == 'batch':
                    x = batch_norm(x, self.is_training)
                elif self.dense_params['normalization'] == 'layer':
                    x = layer_norm(x, self.is_training)
                else:
                    raise ValueError

            if self.is_dropout:
                x = dropout(x, 0.5, self.is_training)
        return dense(x, self.dense_units[-1], activation_='softmax')

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def update_ops(self):
        return [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in ops.name]
