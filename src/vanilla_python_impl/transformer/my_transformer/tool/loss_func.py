try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

from ..model.layer.activation import LogSoftmax


class Loss:
    """Basic class of Loss."""

    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """CrossEntropyLoss."""

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.log_softmax = LogSoftmax()

    def forward(self, y, t):
        """
        shape of y: (num_samples, num_classes)
        shape of t: (num_samples,)
        shape of return: (num_samples,)
        """
        y = np.asarray(y)
        t = np.asarray(t)

        self.old_y = y
        self.old_t = t

        log_softmax = self.log_softmax.forward(y)
        nll_loss = -log_softmax[np.arange(len(t)), t]
        return np.where(t == self.ignore_index, 0, nll_loss)

    def backward(self):
        """
        shape of return: (num_samples, num_classes)
        """
        num_classes = self.old_y.shape[1]
        # mask = np.eye(num_classes).astype(self.old_y.dtype)
        # mask = mask[self.old_t]

        mask = np.zeros((self.old_t.size, num_classes)).astype(self.old_y.dtype)
        mask[np.arange(self.old_t.size), self.old_t] = 1

        grad = np.where(mask == 1, -1, 0).astype(self.old_y.dtype)
        grad = np.where(self.old_t.reshape(-1, 1) == self.ignore_index, 0, grad)

        return self.log_softmax.backward(grad)
