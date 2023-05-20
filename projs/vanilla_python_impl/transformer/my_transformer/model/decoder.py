try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

from .layer import Linear, Embedding, Dropout, DecoderLayer, PositionalEncoding


class Decoder:
    """Transformer Decoder."""

    def __init__(self,
                 tgt_vocab_size,
                 heads_num,
                 layers_num,
                 d_model,
                 d_ff,
                 dropout,
                 max_length=5000,
                 data_type=np.float32):
        self.output_embedding = Embedding(tgt_vocab_size, d_model, data_type)
        self.pe = PositionalEncoding(max_length, d_model, data_type)
        self.fc_out = Linear(in_features=d_model, out_features=tgt_vocab_size, data_type=data_type)
        self.dropout = Dropout(dropout, data_type)
        self.scale = np.sqrt(d_model).astype(data_type)

        self.layers = []
        for _ in range(layers_num):
            self.layers.append(DecoderLayer(d_model, heads_num, d_ff, dropout, data_type))

    def forward(self, tgt, tgt_mask, src, src_mask, training):
        tgt = self.output_embedding.forward(tgt) * self.scale
        tgt = self.pe.forward(tgt)
        tgt = self.dropout.forward(tgt, training)

        for layer in self.layers:
            tgt, attention = layer.forward(tgt, tgt_mask, src, src_mask, training)

        output = self.fc_out.forward(tgt)

        return output, attention

    def backward(self, grad):
        grad = self.fc_out.backward(grad)

        self.encoder_grad = 0
        for layer in reversed(self.layers):
            grad, ecn_grad = layer.backward(grad)
            self.encoder_grad += ecn_grad

        grad = self.dropout.backward(grad)
        grad = self.pe.backward(grad) * self.scale
        grad = self.output_embedding.backward(grad)

    def set_optimizer(self, optimizer):
        self.output_embedding.set_optimizer(optimizer)

        for layer in self.layers:
            layer.set_optimizer(optimizer)

        self.fc_out.set_optimizer(optimizer)

    def update_weights(self):
        layer_num = 1
        self.output_embedding.update_weights(layer_num)

        for layer in self.layers:
            layer_num = layer.update_weights(layer_num)

        self.fc_out.update_weights(layer_num)
