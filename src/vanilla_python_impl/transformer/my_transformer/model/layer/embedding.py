try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class Embedding:
    """
    Embedding layer.
    ---------------
        Args:
            `input_dim`: (int), size of vocabulary
            `output_dim` (int): number of neurons in the layer (vector size)
        Returns:
            input: data with shape (batch_size, input_length)
            output: data with shape (batch_size, input_length, output_dim)
    """

    def __init__(self, input_dim, output_dim, data_type=np.float32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.data_type = data_type

        # used in Adam
        self.num_update = 0

        self._build()

    def _build(self):
        self.w = np.random.normal(0, pow(self.input_dim, -0.5),
                                  (self.input_dim, self.output_dim)).astype(self.data_type)

        self.v, self.m = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)
        self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(
            self.data_type)

    def one_hot(self, batch_labels):
        batch_labels = batch_labels.astype(np.int32)

        prepared_batch_labels = np.zeros((batch_labels.size, self.input_dim))
        prepared_batch_labels[np.arange(batch_labels.size), batch_labels.reshape(1, -1)] = 1

        return prepared_batch_labels.reshape(self.batch_size, self.current_input_length,
                                             self.input_dim).astype(self.data_type)

    def forward(self, X):
        """
        shape of X: (batch_size, input_length)
        X contains values of vocabulary from 0 to input_dim - 1
        """
        self.input_data = X

        if not all([np.equal(len(self.input_data[0]), len(arr)).all() for arr in self.input_data]):
            raise ValueError("Input sequences must be of the same length")

        self.current_input_length = len(self.input_data[0])
        self.batch_size = len(self.input_data)
        self.input_data = self.one_hot(self.input_data)
        self.output_data = np.dot(self.input_data, self.w)

        return self.output_data

    def backward(self, grad):
        """
        shape of grad: (batch_size, input_length, output_dim)
        """
        self.grad_w = np.matmul(np.transpose(self.input_data, axes=(0, 2, 1)), grad).mean(axis=0)

        # output_grad = np.dot(grad, self.w.T)
        # return output_grad
        return None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_weights(self, layer_num):
        self.num_update += 1
        self.w, self.v, self.m, self.v_hat, self.m_hat = self.optimizer.update(self.grad_w, self.w, self.v, self.m,
                                                                               self.v_hat, self.m_hat, self.num_update)

        return layer_num + 1
