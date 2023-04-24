try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class Activation:
    """Basic class of Activation."""

    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Softmax(Activation):
    """Softmax."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        shape of x: (num_samples, num_classes)
        shape of self.old_y: (num_samples, num_classes)
        """
        # self.old_y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        # handle overflow
        t = np.exp(x - x.max(axis=1, keepdims=True))
        self.old_y = t / t.sum(axis=1, keepdims=True)
        return self.old_y

    def backward(self, grad):
        """
        shape of grad: (num_samples, num_classes)
        shape of return: (num_samples, num_classes)
        """
        return self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, None])


class LogSoftmax(Activation):
    """LogSoftmax."""

    def __init__(self):
        self.softmax = Softmax()

    def forward(self, x):
        """
        LogSoftmax forward.

        shape of x: (num_samples, num_classes)
        shape of return: (num_samples, num_classes)
        """
        return np.log(self.softmax.forward(x))

    def backward(self, grad=None):
        """
        shape of grad: (num_samples, num_classes)
        shape of return: (num_samples, num_classes)
        """
        return self.softmax.backward(grad * (1.0 / self.softmax.old_y))


class ReLU(Activation):
    """ReLU."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.old_x = x
        return np.maximum(0, x)

    def backward(self, grad):
        """
        grad shares the same shape as self.old_x
        """
        return grad * np.where(self.old_x <= 0, 0, 1).astype(self.old_x.dtype)
