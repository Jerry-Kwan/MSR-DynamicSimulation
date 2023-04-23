try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class LayerNormalization:
    """
    Applies layer normalization to the input data
    ---------------------------------------------
        Args:
            `epsilon` (float): the epsilon parameter of the algorithm
        Returns:
            output: the normalized input data with same shape

        Reference:
            https://blog.csdn.net/qq_43827595/article/details/121877901
    """

    def __init__(self, normalized_shape=None, epsilon=0.001, data_type=np.float32):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.data_type = data_type

        self.normalized_axis = None

        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None

        self.axis = None

        # used in Adam
        self.num_update = 0

        self._build()

    def _build(self):
        self.feature_size = None

        if self.normalized_shape is not None:
            self.gamma = np.ones(self.normalized_shape).astype(self.data_type)
            self.beta = np.zeros(self.normalized_shape).astype(self.data_type)

            self.vg, self.mg = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(
                self.data_type)
            self.vg_hat, self.mg_hat = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(
                self.gamma).astype(self.data_type)

            self.vb, self.mb = np.zeros_like(self.beta).astype(self.data_type), np.zeros_like(self.beta).astype(
                self.data_type)
            self.vb_hat, self.mb_hat = np.zeros_like(self.beta).astype(self.data_type), np.zeros_like(
                self.beta).astype(self.data_type)

    def forward(self, X):
        self.input_data = X

        # .T: shape(a, b, c, d) -> shape(d, c, b, a)
        x_T = self.input_data.T

        if self.normalized_shape is None:
            self.normalized_shape = self.input_data.shape[1:]
            self._build()

        self.normalized_axis = tuple(np.arange(self.input_data.ndim - self.gamma.ndim).tolist())
        self.feature_size = self.gamma.size

        self.mean = np.mean(x_T, axis=0)
        self.var = np.var(x_T, axis=0)

        self.X_centered = (x_T - self.mean)
        self.stddev_inv = 1.0 / np.sqrt(self.var + self.epsilon)

        self.X_hat_T = self.X_centered * self.stddev_inv
        self.X_hat = self.X_hat_T.T

        self.output_data = self.gamma * self.X_hat + self.beta

        return self.output_data

    def backward(self, grad):
        """
        output_grad has the same shape as grad
        """
        batch_size = grad.shape[0]

        grad_T = grad.T

        output_grad = (1 / self.feature_size) * np.expand_dims(
            self.gamma, axis=self.normalized_axis).T * self.stddev_inv * (
                self.feature_size * grad_T - np.sum(grad_T, axis=0) -
                self.X_centered * np.power(self.stddev_inv, 2) * np.sum(grad_T * self.X_centered, axis=0))

        output_grad = output_grad.T

        self.grad_gamma = np.sum(grad * self.X_hat, axis=self.normalized_axis) / batch_size
        self.grad_beta = np.sum(grad, axis=self.normalized_axis) / batch_size

        return output_grad

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_weights(self, layer_num):
        self.num_update += 1

        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat = self.optimizer.update(
            self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, self.num_update)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(
            self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, self.num_update)

        return layer_num + 1
