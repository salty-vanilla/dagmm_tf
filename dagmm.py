import tensorflow as tf
from tensorflow.python.keras.layers import Input
import numpy as np
import os
import csv
import time
from PIL import Image
from models import AutoEncoder, EstimationNetwork, GMM
from ops.losses.gan import generator_loss, \
    discriminator_loss, \
    gradient_penalty, \
    discriminator_norm


class DAGMM:
    def __init__(self, autoencoder,
                 estimation_network,
                 gmm,
                 lambda1,
                 lambda2,
                 learning_rate):
        self.autoencoder = autoencoder
        self.estimation_network = estimation_network
        self.gmm = gmm

        with tf.Graph().as_default():
            self.x = Input(shape=self.autoencoder.input_shape,
                           dtype=tf.float32)

            # train
            self.encoded = self.autoencoder.encode(self.x,
                                                   is_training=True,
                                                   reuse=False)
            self.decoded = self.autoencoder.decode(self.encoded,
                                                   is_training=True,
                                                   reuse=False)
            self.distance = self.autoencoder.calc_distance(self.x,
                                                           self.decoded,
                                                           is_training=True,
                                                           reuse=False)

            self.z = tf.concat([self.encoded, self.distance], axis=-1)

            self.gamma = self.estimation_network(self.z,
                                                 is_training=True,
                                                 reuse=False)

            self.gmm.build(self.z, self.gamma)
            self.energy = self.gmm.calc_energy()



