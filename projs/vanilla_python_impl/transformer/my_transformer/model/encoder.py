try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

from .layer import Embedding, Dropout, EncoderLayer, PositionalEncoding


class Encoder:
    """Transformer Encoder."""

    def __init__(self,
                 src_vocab_size,
                 heads_num,
                 layers_num,
                 d_model,
                 d_ff,
                 dropout,
                 max_length=5000,
                 data_type=np.float32):
        self.input_embedding = Embedding(src_vocab_size, d_model, data_type)
        self.pe = PositionalEncoding(max_length, d_model, data_type)
        self.dropout = Dropout(dropout, data_type)
        self.scale = np.sqrt(d_model).astype(data_type)

        self.layers = []
        for _ in range(layers_num):
            self.layers.append(EncoderLayer(d_model, heads_num, d_ff, dropout, data_type))

    def forward(self, src, src_mask, training):
        src = self.input_embedding.forward(src) * self.scale
        src = self.pe.forward(src)
        src = self.dropout.forward(src, training)

        for layer in self.layers:
            src = layer.forward(src, src_mask, training)

        return src

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        grad = self.dropout.backward(grad)
        grad = self.pe.backward(grad) * self.scale
        grad = self.input_embedding.backward(grad)

    def set_optimizer(self, optimizer):
        self.input_embedding.set_optimizer(optimizer)

        for layer in self.layers:
            layer.set_optimizer(optimizer)

    def update_weights(self):
        layer_num = 1
        layer_num = self.input_embedding.update_weights(layer_num)

        for layer in self.layers:
            layer_num = layer.update_weights(layer_num)
