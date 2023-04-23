try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

from .linear import Linear
from .dropout import Dropout
from .activation import Softmax


class MultiHeadAttention:
    """Multi-Head Attention."""

    def __init__(self, d_model=512, heads_num=8, dropout=0.1, data_type=np.float32):
        self.d_model = d_model
        self.heads_num = heads_num
        self.dropout = dropout
        self.data_type = data_type

        self._build()

    def _build(self):
        self.d_k = self.d_q = self.d_v = self.d_model // self.heads_num
        self.scale = np.sqrt(self.d_k).astype(self.data_type)

        self.activation = Softmax()
        self.dropout = Dropout(self.dropout)

        self.W_q = Linear(in_features=self.d_model,
                          out_features=self.d_q * self.heads_num,
                          bias=False,
                          data_type=self.data_type)
        self.W_k = Linear(in_features=self.d_model,
                          out_features=self.d_k * self.heads_num,
                          bias=False,
                          data_type=self.data_type)
        self.W_v = Linear(in_features=self.d_model,
                          out_features=self.d_v * self.heads_num,
                          bias=False,
                          data_type=self.data_type)
        self.W_o = Linear(in_features=self.d_v * self.heads_num,
                          out_features=self.d_model,
                          bias=True,
                          data_type=self.data_type)

    def split_heads_forward(self, x, dim):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.heads_num, dim).transpose(0, 2, 1, 3)

    def group_heads_forward(self, x):
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * self.d_v)

    def forward(self, query, key, value, mask, training=True):
        """
        shape of qkv: (batch_size, q/k/v_length, d_model)
        shape of mask:
            (batch_size, ?, seq_len) - ? is 1 for src mask, seq_len for tgt mask
            the i-th row of mask(seq_len, seq_len) represents mask of the i-th word to other words
        """
        # shape after q/k/v_linear: (batch_size, q/k/v_length, self.d_q/k/v * heads_num)
        # shape after split: (batch_size, self.heads_num, q/k/v_length, self.d_q/k/v)
        self.Q = self.split_heads_forward(self.W_q.forward(query), self.d_q)
        self.K = self.split_heads_forward(self.W_k.forward(key), self.d_k)
        self.V = self.split_heads_forward(self.W_v.forward(value), self.d_v)

        # shape of energy: (batch_size, self.heads_num, q/k_length, q/k_length)
        energy = np.matmul(self.Q, self.K.transpose(0, 1, 3, 2)) / self.scale

        # add mask
        self.mask = np.asarray(mask)
        if self.mask is not None:
            # shape after newaxis: (batch_size, 1, ?, seq_len)
            # ? is 1 for src mask, seq_len for tgt mask
            self.mask = self.mask[:, np.newaxis, ...]
            energy = np.where(self.mask == 0, float('-inf'), energy)

        # softmax activation
        qk_len = energy.shape[3]
        energy = energy.reshape(-1, qk_len)
        attention = self.activation.forward(energy)
        attention = attention.reshape(-1, self.heads_num, qk_len, qk_len)

        # QK^TV, shape of output is (batch_size, self.heads_num, q/k_length, self.d_v)
        self.dropout_attention = self.dropout.forward(attention, training)
        output = np.matmul(self.dropout_attention, self.V)

        # shape of concat_output: (batch_size, self.query_len, self.heads_num * self.d_v)
        concat_output = self.group_heads_forward(output)
        return self.W_o.forward(concat_output), attention

    def group_heads_backward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.heads_num, self.d_v).transpose(0, 2, 1, 3)

    def split_heads_backward(self, x, dim):
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * dim)

    def backward(self, grad):
        # shape of grad: (batch_size, self.query_len, d_model)
        # shape after W_o: (batch_size, self.query_len, self.heads_num * self.d_v)
        grad = self.W_o.backward(grad)

        # shape after group back: (batch_size, self.heads_num, seq_len, self.d_v)
        grad = self.group_heads_backward(grad)

        # same as embedding backward, see my notes for details
        # shape of V_grad: (batch_size, self.heads_num, seq_len, self.d_v)
        V_grad = np.matmul(self.dropout_attention.transpose(0, 1, 3, 2), grad)

        # shape of output grad: (batch_size, self.heads_num, seq_len, seq_len)
        grad = np.matmul(grad, self.V.transpose(0, 1, 3, 2))
        grad = self.dropout.backward(grad)

        qk_len = grad.shape[3]
        grad = grad.reshape(-1, qk_len)
        grad = self.activation.backward(grad)
        grad = grad.reshape(-1, self.heads_num, qk_len, qk_len)

        if self.mask is not None:
            grad = np.where(self.mask == 0, 0, grad)

        grad /= self.scale

        # see my notes for embedding grad for details
        # shape of Q/K_grad: (batch_size, self.heads_num, qk_len, d_q/k)
        Q_grad = np.matmul(grad, self.K)
        K_grad = np.matmul(self.Q.transpose(0, 1, 3, 2), grad)
        K_grad = K_grad.transpose(0, 1, 3, 2)

        # shape after split back: (batch_size, qk_len, self.heads_num * d_q/k/v)
        Q_grad = self.split_heads_backward(Q_grad, self.d_q)
        K_grad = self.split_heads_backward(K_grad, self.d_k)
        V_grad = self.split_heads_backward(V_grad, self.d_v)

        Q_grad = self.W_q.backward(Q_grad)
        K_grad = self.W_k.backward(K_grad)
        V_grad = self.W_v.backward(V_grad)

        # shape: (batch_size, qk_len, d_model)
        return Q_grad, K_grad, V_grad

    def set_optimizer(self, optimizer):
        self.W_q.set_optimizer(optimizer)
        self.W_k.set_optimizer(optimizer)
        self.W_v.set_optimizer(optimizer)
        self.W_o.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.W_k.update_weights(layer_num)
        layer_num = self.W_q.update_weights(layer_num)
        layer_num = self.W_v.update_weights(layer_num)
        layer_num = self.W_o.update_weights(layer_num)

        return layer_num
