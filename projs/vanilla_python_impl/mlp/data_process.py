import os
import numpy as np
from urllib import request
import gzip
import pickle
import matplotlib.pyplot as plt
import random

filename = [['training_images', 'train-images-idx3-ubyte.gz'], ['test_images', 't10k-images-idx3-ubyte.gz'],
            ['training_labels', 'train-labels-idx1-ubyte.gz'], ['test_labels', 't10k-labels-idx1-ubyte.gz']]


def download_mnist(path):
    """Download MNIST dataset."""
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    for name in filename:
        print('Downloading ' + name[1] + '...')
        request.urlretrieve(base_url + name[1], os.path.join(path, name[1]))

    print('Download complete.')


def save_mnist(path):
    """Save MNIST dataset to one file."""
    mnist = {}

    # image
    for name in filename[:2]:
        with gzip.open(os.path.join(path, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

    # label
    for name in filename[-2:]:
        with gzip.open(os.path.join(path, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    # save as one file
    with open(os.path.join(path, 'mnist.pkl'), 'wb') as f:
        pickle.dump(mnist, f)

    print('Save complete.')


def load_data(path, download=False, process_mnist=False):
    """Load MNIST dataset from mnist.pkl file."""
    if download:
        download_mnist(path)

    if process_mnist:
        save_mnist(path)

    with open(os.path.join(path, 'mnist.pkl'), 'rb') as f:
        mnist = pickle.load(f)

    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']


def one_hot(num_classes, x):
    """One-hot encoding."""
    # x.shape = (n,)
    return np.eye(num_classes)[x]


def data_iter(x, y, batch_size, shuffle=True):
    """Return a data iterator on (x, y)."""
    idx = list(range(len(x)))

    if shuffle:
        random.shuffle(idx)

    num_batches = len(x) // batch_size

    for i in range(num_batches):
        yield x[idx[i * batch_size:(i + 1) * batch_size]], y[idx[i * batch_size:(i + 1) * batch_size]]


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_data('mlp_mnist/', download=False, process_mnist=False)
    print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
    print(x_train[0][:10])
    print(t_train[:10])

    img = x_train[0, :].reshape(28, 28)  # First image in the training set.
    plt.imshow(img, cmap='gray')
    plt.show()  # Show the image
