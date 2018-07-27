# I referred to https://github.com/tnakae/DAGMM/blob/master/dagmm/gmm.py

import numpy as np
import tensorflow as tf


class GMM:
    def __init__(self, nb_components,
                 nb_features):
        self.nb_components = nb_components
        self.nb_features = nb_features
        self.is_training = False

        with tf.variable_scope("GMM"):
            self.phi = self.mu = self.sigma = self.L = None
            self._phi = self._mu = self._sigma = self._L = None
            self._create_vars()

    def _create_vars(self):
        self.phi = tf.Variable(tf.zeros(shape=(self.nb_components, )),
                               dtype=tf.float32,
                               name='phi')
        self.mu = tf.Variable(tf.zeros(shape=(self.nb_components,
                                              self.nb_features)),
                              dtype=tf.float32,
                              name='mu')
        self.sigma = tf.Variable(tf.zeros(shape=(self.nb_components,
                                                 self.nb_features,
                                                 self.nb_features)),
                                 dtype=tf.float32,
                                 name='sigma')
        self.L = tf.Variable(tf.zeros(shape=(self.nb_components,
                                             self.nb_features,
                                             self.nb_features)),
                             dtype=tf.float32,
                             name='L')

    def build(self, z,
              gamma):
        with tf.variable_scope("GMM"):
            # Calculate mu, sigma
            # i   : index of samples
            # k   : index of components
            # l,m : index of features
            gamma_sum = tf.reduce_sum(gamma, axis=0)
            self._phi = tf.reduce_mean(gamma, axis=0)
            self._mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, None]
            z_centered = tf.sqrt(gamma[:, :, None]) * (z[:, None, :] - self._mu[None, :, :])
            self._sigma = tf.einsum('ikl,ikm->klm',
                                    z_centered,
                                    z_centered)
            self._sigma /= gamma_sum[:, None, None]

            # Calculate a cholesky decomposition of covariance in advance
            n_features = z.shape[1]
            min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
            self._L = tf.cholesky(self._sigma + min_vals[None, :, :])

    def calc_energy(self, z,
                    is_training=True):
        if is_training:
            phi = self._phi
            mu = self._mu
            L = self._L
        else:
            phi = self.phi
            mu = self.mu
            L = self.L

        with tf.variable_scope("GMM_energy"):
            # Instead of inverse covariance matrix, exploit cholesky decomposition
            # for stability of calculation.
            z_centered = z[:, None, :] - mu[None, :, :]  #ikl
            v = tf.matrix_triangular_solve(L,
                                           tf.transpose(z_centered, [1, 2, 0]))  # kli

            # log(det(Sigma)) = 2 * sum[log(diag(L))]
            log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)

            # To calculate energies, use "log-sum-exp" (different from orginal paper)
            d = z.get_shape().as_list()[1]
            logits = tf.log(phi[:, None]) \
                     - 0.5*(tf.reduce_sum(tf.square(v), axis=1)
                            + d*tf.log(2.0 * np.pi) + log_det_sigma[:, None])
            energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies

    def cov_diag_loss(self, is_training=True):
        if is_training:
            sigma = self._sigma
        else:
            sigma = self.sigma
        return tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(sigma)))

    @property
    def update_ops(self):
        return tf.group(tf.assign(self.phi, self._phi),
                        tf.assign(self.mu, self._mu),
                        tf.assign(self.sigma, self._sigma),
                        tf.assign(self.L, self._L))
