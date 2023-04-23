try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class PositionalEncoding:
    """
    Implementation of Positional Encoding.

    Reference:
        https://www.cnblogs.com/xiximayou/p/13343665.html
    """

    def __init__(self, max_len, d_model, data_type=np.float32):
        self.d_model = d_model
        self.max_len = max_len
        self.data_type = data_type

        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]  # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model / 2,)

        pe[:, 0::2] = np.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])

        self.pe = pe[np.newaxis, :, :].astype(self.data_type)  # (1, max_len, d_model)

    def forward(self, x):
        """
        shape of return is the same as the shape of x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.shape[1]]

    def backward(self, grad):
        """
        shape of grad: (batch_size, seq_len, d_model)
        """
        return grad
