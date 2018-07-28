import numpy as np
from tensorflow.python.keras.datasets import mnist


def get_digits_per_label(label, nb_sampling=None, phase='train'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    assert phase in ['train', 'test']

    x = x_train if phase == 'train' else x_test
    y = y_train if phase == 'train' else y_test

    x_ = x[y == label]

    if nb_sampling is None:
        return x_
    else:
        return x_[:nb_sampling]


def get_digits(labels, nb_sampling_list, phase='train', with_label=False):
    x = np.empty((0, 28, 28))
    y = np.empty((0, ))
    for label, nb_sampling in zip(labels, nb_sampling_list):
        _x = get_digits_per_label(label, nb_sampling, phase)
        x = np.append(x, _x, axis=0)
        y = np.append(y, np.array([label for _ in range(len(_x))]))

    if with_label:
        return x, y
    else:
        return x


if __name__ == '__main__':
    print(get_digits([0, 1], [None, None], 'test').shape)