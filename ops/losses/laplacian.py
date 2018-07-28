import tensorflow as tf
from ops.losses.utils import make_gauss_kernel


def make_gaussian_pyramid(x,
                          max_level=5,
                          kernel_size=(3, 3),
                          sigma=1.0,
                          gaussian_iteration=1):
    bs, h, w, c = x.get_shape().as_list()
    pyramid = [x]
    g_kernel = make_gauss_kernel(kernel_size, sigma)
    g_kernel = tf.tile(g_kernel, (1, 1, c, 1))

    for level in range(max_level):
        current = pyramid[-1]
        downsampled = tf.nn.avg_pool(current, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        filtered = downsampled
        for _ in range(gaussian_iteration):
            filtered = tf.nn.depthwise_conv2d(filtered, g_kernel,
                                              strides=[1, 1, 1, 1], padding='SAME')
        pyramid.append(filtered)
    return pyramid


def make_laplacian_pyramid(x,
                           max_level=5,
                           kernel_size=(3, 3),
                           sigma=1.0,
                           gaussian_iteration=1):
    g_pyr = make_gaussian_pyramid(x, max_level, kernel_size, sigma, gaussian_iteration)
    l_pyr = []
    for level in range(max_level):
        high_reso = g_pyr[level]
        low_reso = g_pyr[level + 1]

        bs, h, w, c = high_reso.get_shape().as_list()
        up_low_reso = tf.image.resize_bilinear(low_reso, size=(w, h))

        diff = high_reso - up_low_reso
        l_pyr.append(diff)
    return l_pyr


def lap1_loss(y_true,
              y_pred,
              max_level=5,
              kernel_size=(3, 3),
              sigma=1.0,
              gaussian_iteration=1):
    true_pyr = make_laplacian_pyramid(y_true,
                                      max_level=max_level,
                                      kernel_size=kernel_size,
                                      sigma=sigma,
                                      gaussian_iteration=gaussian_iteration)
    pred_pyr = make_laplacian_pyramid(y_pred,
                                      max_level=max_level,
                                      kernel_size=kernel_size,
                                      sigma=sigma,
                                      gaussian_iteration=gaussian_iteration)

    diffs = []
    for t, p in zip(true_pyr, pred_pyr):
        d = tf.reduce_mean(tf.abs(t - p), axis=[1, 2, 3])
        diffs.append(tf.expand_dims(d, axis=-1))
    diffs = tf.concat(diffs, axis=-1)
    loss = tf.reduce_mean(diffs, axis=-1)
    return loss
