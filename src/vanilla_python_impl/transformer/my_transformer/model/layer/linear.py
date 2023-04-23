try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class Linear:
    """
    Add Dense layer
    ---------------
        Args:
            in_features (int): dimension of the input
            out_features (int): dimension of the output
            bias (bool): True if used. False if not used
        Returns:
            output: data with shape (batch_size, out_features)
    """

    def __init__(self, in_features, out_features, bias=True, data_type=np.float32):
        self.out_features = out_features
        self.in_features = in_features
        self.bias = bias
        self.data_type = data_type

        # used in Adam
        self.num_update = 0

        self.build()

    def build(self):
        # self.w = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.out_features))

        # xavier initialization
        stdv = 1. / np.sqrt(self.in_features)  # * 0.5 # input size
        # stdv = np.sqrt(6) / np.sqrt(self.input_size + self.out_features)
        # kaiming initialization
        # stdv = np.sqrt(2 / self.input_size)

        self.w = np.random.uniform(-stdv, stdv, (self.in_features, self.out_features)).astype(self.data_type)

        # if self.bias == True:
        #     self.b = np.random.uniform(-stdv, stdv, self.out_features)
        # else:
        #     self.b = np.zeros(self.out_features)
        self.b = np.zeros(self.out_features).astype(self.data_type)

        # optimizer params
        self.v, self.m = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(
            self.data_type)
        self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(
            self.data_type)

        # optimizer params
        self.vb, self.mb = np.zeros_like(self.b).astype(self.data_type), np.zeros_like(self.b).astype(
            self.data_type)
        self.vb_hat, self.mb_hat = np.zeros_like(self.b).astype(self.data_type), np.zeros_like(self.b).astype(
            self.data_type)

    def forward(self, X, training=True):
        """
        shape of X is (batch_size, ..., in_features)
        shape of self.output_data is (batch_size, ..., out_features)
        """
        self.input_data = X
        self.batch_size = len(self.input_data)
        self.output_data = np.dot(self.input_data, self.w) + self.b

        return self.output_data

    def backward(self, grad):
        """
        shape of grad is the same as the output of forward method, that is (batch_size, ..., out_features)
        """
        self.grad_w = np.mean(np.matmul(self.input_data.transpose(0, 2, 1), grad), axis=0)
        self.grad_b = np.mean(grad, axis=(0, 1))

        return np.dot(grad, self.w.T)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_weights(self, layer_num):
        self.num_update += 1
        self.w, self.v, self.m, self.v_hat, self.m_hat = self.optimizer.update(self.grad_w, self.w, self.v, self.m,
                                                                               self.v_hat, self.m_hat, self.num_update)

        if self.bias:
            self.b, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(
                self.grad_b, self.b, self.vb, self.mb, self.vb_hat, self.mb_hat, self.num_update)

        return layer_num + 1
