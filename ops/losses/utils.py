import tensorflow as tf
import numpy as np


def make_gauss_kernel(kernel_size, sigma):
    _y, _x = np.mgrid[-kernel_size[1] // 2 + 1: kernel_size[1] // 2 + 1,
                      -kernel_size[0] // 2 + 1: kernel_size[0] // 2 + 1]

    _x = _x.reshape(list(_x.shape) + [1, 1])
    _y = _y.reshape(list(_y.shape) + [1, 1])
    x = tf.constant(_x, dtype=tf.float32)
    y = tf.constant(_y, dtype=tf.float32)

    g = tf.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g /= tf.reduce_sum(g)
    return g
