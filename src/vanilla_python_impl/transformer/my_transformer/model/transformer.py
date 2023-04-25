try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

import os
import pickle

from .encoder_decoder import EncoderDecoder
from .encoder import Encoder
from .decoder import Decoder


class Transformer(EncoderDecoder):
    """
    My implementation of Transformer.

    References:
        1. Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf)
        2. https://github.com/AkiRusProd/numpy-transformer
        3. https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        4. https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html
        5. https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html
        6. https://www.cnblogs.com/xiximayou/p/13343665.html
        7. ...
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 src_max_len,
                 tgt_max_len,
                 num_enc_heads=8,
                 num_dec_heads=8,
                 num_enc_layers=6,
                 num_dec_layers=6,
                 d_model=512,
                 d_ff=2048,
                 enc_dropout=0.1,
                 dec_dropout=0.1,
                 data_type=np.float32):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.num_enc_heads = num_enc_heads
        self.num_dec_heads = num_dec_heads

        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        self.d_model = d_model
        self.d_ff = d_ff

        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout

        self.data_type = data_type

        self.optim_set = False

        self._build()

    def _build(self):
        self.encoder = Encoder(self.src_vocab_size, self.num_enc_heads, self.num_enc_layers, self.d_model, self.d_ff,
                               self.enc_dropout, self.src_max_len, self.data_type)

        self.decoder = Decoder(self.tgt_vocab_size, self.num_dec_heads, self.num_dec_layers, self.d_model, self.d_ff,
                               self.dec_dropout, self.tgt_max_len, self.data_type)

    def set_optimizer(self, optimizer):
        self.encoder.set_optimizer(optimizer)
        self.decoder.set_optimizer(optimizer)
        self.optim_set = True

    def update_weights(self):
        assert self.optim_set is True, 'please set optimizer before updating weights'
        self.encoder.update_weights()
        self.decoder.update_weights()

    def load_model(self, path, suffix):
        pickle_encoder = open(f'{path}/encoder-{suffix}.pkl', 'rb')
        pickle_decoder = open(f'{path}/decoder-{suffix}.pkl', 'rb')

        self.encoder = pickle.load(pickle_encoder)
        self.decoder = pickle.load(pickle_decoder)

        pickle_encoder.close()
        pickle_decoder.close()

        print(f'---load from "{path}" with suffix {suffix}---', end='\n\n')

    def save_model(self, path, suffix):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_encoder = open(f'{path}/encoder-{suffix}.pkl', 'wb')
        pickle_decoder = open(f'{path}/decoder-{suffix}.pkl', 'wb')

        pickle.dump(self.encoder, pickle_encoder)
        pickle.dump(self.decoder, pickle_decoder)

        pickle_encoder.close()
        pickle_decoder.close()

        print(f'---save to "{path}" with suffix {suffix}---', end='\n\n')

    def forward(self, src, tgt, src_mask, tgt_mask, training):
        """
        Shape:
            src - (batch_size, src_seq_len)
            tgt - (batch_size, tgt_seq_len)
            src_mask - (batch_size, 1, src_seq_len)
            tgt_mask - (batch_size, tgt_seq_len, tgt_seq_len)
            out - (batch_size, tgt_seq_len, tgt_vocab_size)
            attn: (batch_size, num_dec_heads, tgt_seq_len, src_seq_len)
        """
        src, tgt = src.astype(self.data_type), tgt.astype(self.data_type)
        enc_src = self.encoder.forward(src, src_mask, training)
        out, attn = self.decoder.forward(tgt, tgt_mask, enc_src, src_mask, training)
        return out, attn

    def backward(self, grad):
        grad = self.decoder.backward(grad)
        grad = self.encoder.backward(self.decoder.encoder_grad)
