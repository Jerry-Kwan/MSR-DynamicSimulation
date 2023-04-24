try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

from .dropout import Dropout
from .multi_head_attn import MultiHeadAttention
from .positionwise_ffn import PositionwiseFFN
from .layer_norm import LayerNorm


class DecoderLayer:
    "One of Transformer Decoder Layer."

    def __init__(self, d_model, heads_num, d_ff, dropout, data_type=np.float32):
        self.self_attn = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.enc_attn = MultiHeadAttention(d_model, heads_num, dropout, data_type)

        self.ln1 = LayerNorm(d_model, epsilon=1e-6, data_type=data_type)
        self.ln2 = LayerNorm(d_model, epsilon=1e-6, data_type=data_type)
        self.ln3 = LayerNorm(d_model, epsilon=1e-6, data_type=data_type)

        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.dropout = Dropout(dropout, data_type)

    def forward(self, tgt, tgt_mask, src, src_mask, training):
        # sub1: self attn forward
        self_attn_out, _ = self.self_attn.forward(tgt, tgt, tgt, tgt_mask, training)
        sub1_out = self.ln1.forward(tgt + self.dropout.forward(self_attn_out, training))

        # sub2: enc attn forward
        enc_attn_out, attention = self.enc_attn.forward(sub1_out, src, src, src_mask, training)
        sub2_out = self.ln2.forward(sub1_out + self.dropout.forward(enc_attn_out, training))

        # sub3: ffn forward
        ffn_out = self.ffn.forward(sub2_out, training)
        sub3_out = self.ln3.forward(ffn_out + self.dropout.forward(ffn_out, training))

        return sub3_out, attention

    def backward(self, grad):
        grad = self.ln3.backward(grad)

        _grad = self.ffn.backward(self.dropout.backward(grad))
        grad = self.ln2.backward(grad + _grad)

        _grad, enc_grad1, enc_grad2 = self.enc_attn.backward(self.dropout.backward(grad))
        grad = self.ln1.backward(grad + _grad)

        _grad, _grad2, _grad3 = self.self_attn.backward(self.dropout.backward(grad))

        # the first one is grad for decoder, the second one is grad for encoder
        return _grad + _grad2 + _grad3 + grad, enc_grad1 + enc_grad2

    def set_optimizer(self, optimizer):
        self.self_attn.set_optimizer(optimizer)
        self.enc_attn.set_optimizer(optimizer)

        self.ln1.set_optimizer(optimizer)
        self.ln2.set_optimizer(optimizer)
        self.ln3.set_optimizer(optimizer)

        self.ffn.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.ln1.update_weights(layer_num)
        layer_num = self.ln2.update_weights(layer_num)
        layer_num = self.ln3.update_weights(layer_num)
        layer_num = self.self_attn.update_weights(layer_num)
        layer_num = self.enc_attn.update_weights(layer_num)
        layer_num = self.ffn.update_weights(layer_num)

        return layer_num
