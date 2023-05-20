try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

from .multi_head_attn import MultiHeadAttention
from .layer_norm import LayerNorm
from .dropout import Dropout
from .positionwise_ffn import PositionwiseFFN


class EncoderLayer:
    """One of Transformer Encoder Layer."""

    def __init__(self, d_model, heads_num, d_ff, dropout, data_type=np.float32):
        self.attn = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

        self.ln1 = LayerNorm(d_model, epsilon=1e-6, data_type=data_type)
        self.ln2 = LayerNorm(d_model, epsilon=1e-6, data_type=data_type)

        self.dropout = Dropout(dropout, data_type)

    def forward(self, src, src_mask, training):
        attn_out, _ = self.attn.forward(src, src, src, src_mask, training)
        sub1_out = self.ln1.forward(src + self.dropout.forward(attn_out, training))

        ffn_out = self.ffn.forward(sub1_out, training)
        sub2_out = self.ln2.forward(sub1_out + self.dropout.forward(ffn_out, training))

        return sub2_out

    def backward(self, grad):
        # back of sub2
        grad = self.ln2.backward(grad)
        _grad = self.ffn.backward(self.dropout.backward(grad))

        # back of sub1
        grad = self.ln1.backward(grad + _grad)
        _grad, _grad2, _grad3 = self.attn.backward(self.dropout.backward(grad))

        # according to chain rules, return the sum of grads
        return _grad + _grad2 + _grad3 + grad

    def set_optimizer(self, optimizer):
        self.ln1.set_optimizer(optimizer)
        self.ln2.set_optimizer(optimizer)
        self.attn.set_optimizer(optimizer)
        self.ffn.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.ln1.update_weights(layer_num)
        layer_num = self.ln2.update_weights(layer_num)
        layer_num = self.attn.update_weights(layer_num)
        layer_num = self.ffn.update_weights(layer_num)

        return layer_num
