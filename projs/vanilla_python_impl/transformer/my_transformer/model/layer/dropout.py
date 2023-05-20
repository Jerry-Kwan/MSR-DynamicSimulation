try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class Dropout:
    """Implementation of Dropout."""

    def __init__(self, rate=0.1, dtype=np.float32):
        self.rate = rate
        self.dtype = dtype

    def forward(self, X, training=True):
        """
        Return the dropout result, whose shape is the same as the input.

        Reference:
            https://blog.csdn.net/scar2016/article/details/123658920
        """
        if not training:
            return X

        self.mask = np.random.binomial(n=1, p=1 - self.rate, size=X.shape).astype(self.dtype)
        return X * self.mask / (1.0 - self.rate)

    def backward(self, error):
        return error * self.mask / (1.0 - self.rate)
