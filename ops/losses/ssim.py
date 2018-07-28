import tensorflow as tf
from ops.losses.utils import make_gauss_kernel


def calc_ssim(y_true,
              y_pred,
              L=1.0,
              K1=0.01,
              K2=0.03,
              kernel_size=(3, 3),
              sigma=1.0):
    """SSIM
    paper: https://ece.uwaterloo.ca/~z70wang/research/ssim/
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        L: float hyper parameter of SSIM.
        K1: float hyper parameter of SSIM.
        K2: float hyper parameter of SSIM.
        kernel_size: size of gaussian kernel. (x, y)
        sigma: float parameter of gaussian.
    # Returns
        4D Tensor (None, h-p, w-p, c).
        Each element of the tensor represents SSIM .
    """

    bs, h, w, c = y_true.get_shape().as_list()

    g_kernel = make_gauss_kernel(kernel_size, sigma)
    g_kernel = tf.tile(g_kernel, (1, 1, c, 1))

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_true = tf.nn.depthwise_conv2d(y_true, g_kernel, strides=[1, 1, 1, 1], padding='VALID')
    mu_pred = tf.nn.depthwise_conv2d(y_pred, g_kernel, strides=[1, 1, 1, 1], padding='VALID')

    mu_true_true = mu_true * mu_true
    mu_pred_pred = mu_pred * mu_pred
    mu_true_pred = mu_true * mu_pred

    sigma_true_true = tf.nn.conv2d(y_true * y_true, g_kernel,
                                   strides=[1, 1, 1, 1], padding='VALID') - mu_true_true
    sigma_pred_pred = tf.nn.conv2d(y_pred * y_pred, g_kernel,
                                   strides=[1, 1, 1, 1], padding='VALID') - mu_pred_pred
    sigma_true_pred = tf.nn.conv2d(y_true * y_pred, g_kernel,
                                   strides=[1, 1, 1, 1], padding='VALID') - mu_true_pred

    ssim = (2 * mu_true_pred + C1) * (2 * sigma_true_pred + C2)
    ssim /= (mu_true_true + mu_pred_pred + C1) * (sigma_true_true + sigma_pred_pred + C2)
    return ssim


def ssim_loss(y_true,
              y_pred,
              L=1.0,
              K1=0.01,
              K2=0.03,
              kernel_size=(3, 3),
              sigma=1.0):
    """SSIM loss function
    paper: https://ece.uwaterloo.ca/~z70wang/research/ssim/
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        L: float hyper parameter of SSIM.
        K1: float hyper parameter of SSIM.
        K2: float hyper parameter of SSIM.
        kernel_size: size of gaussian kernel. (x, y)
        sigma: float parameter of gaussian.
    # Returns
        Tensor with one scalar loss entry per sample.
        Each scalar represents "1 - ssim" .
    """

    ssim = calc_ssim(y_true, y_pred,
                     L, K1, K2,
                     kernel_size, sigma)
    return 1. - tf.reduce_mean(ssim, axis=[1, 2, 3])