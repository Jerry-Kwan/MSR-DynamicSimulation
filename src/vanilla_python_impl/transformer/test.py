import torch
import numpy as np
from my_transformer import Adam, Noam, CrossEntropyLoss, PositionalEncoding, Embedding, Linear, PositionwiseFFN, MultiHeadAttention, LayerNorm, EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer

# a = CrossEntropyLoss()
# x_input = torch.randn(4, 3)  #随机生成输入
# y_target = torch.tensor([1, 2, 0, 1])  #设置输出具体值 print('y_target\n',y_target)
# print(np.mean(a.forward(x_input, y_target)))
# print(a.backward())
# print(a.log_softmax.softmax)
# x_input.requires_grad_()
# crossentropyloss = torch.nn.CrossEntropyLoss()
# crossentropyloss_output = crossentropyloss(x_input, y_target)
# print('crossentropyloss_output:\n', crossentropyloss_output)
# crossentropyloss_output.backward()
# print(x_input.grad * 4)

# a = PositionalEncoding(50, 4)
# print(a.forward(np.zeros((2, 3, 4))))

# a = Embedding(200, 3)
# print(a.forward(np.arange(24).reshape(3, 8)))
# print('sssssssssss')
# print(a.backward(np.random.randn(3, 8, 3)).shape)

# a = Linear(3, 2)
# print(a.forward(np.random.randn(4, 2, 2)))
# print(a.backward(np.random.randn(4, 2, 3)))

# a = PositionwiseFFN(3, 2)
# print(a.forward(np.random.randn(2, 4, 3)))
# print(a.backward(np.random.randn(2, 4, 3)))

# a = MultiHeadAttention(4, 2, 0.1)
# k = np.random.randn(2, 3, 4)
# print(a.forward(k, k, k, np.ones((2, 1, 3)), training=True)[0])
# print(a.backward(np.random.randn(2, 3, 4)))

# a = LayerNorm(5)
# print(np.sum(a.forward(np.random.randn(3, 4, 5)), axis=-1))
# print(a.backward(np.random.randn(3, 4, 5)))

# a = EncoderLayer(4, 2, 3, 0.1)
# src = np.random.randn(3, 3, 4)
# print(a.forward(src, np.ones((3, 3, 3)), True))
# print(a.backward(np.ones_like(src)))

# a = DecoderLayer(2, 2, 2, 0.1)
# src = np.random.randn(2, 2, 2)
# tgt = np.random.randn(2, 3, 2)
# print(a.forward(tgt, np.ones((2, 3, 3)), src, np.ones((2, 1, 2)), True)[0].shape)
# print([x.shape for x in a.backward(np.ones_like(tgt))])

# a = Encoder(5, 2, 2, 3, 3, 0.1, 10)
# src = np.array([
#     [4, 2, 3, 0, 1],
#     [3, 4, 1, 0, 2]
# ])
# print(a.forward(src, np.ones((2, 1, 5)), True))
# print(a.backward(np.ones((2, 5, 3))))

# a = Decoder(5, 2, 2, 3, 2, 0.1, 10)
# tgt = np.array([
#     [4, 2, 3, 0, 1],
#     [3, 4, 1, 0, 2]
# ])
# print(a.forward(tgt, np.ones((2, 5, 5)), np.random.randn(2, 3, 3), np.ones((2, 1, 3)), True))
# print(a.backward(np.ones((2, 5, 5))))

# vocab_size = 10
# max_len = 10
# num_heads = 2
# num_layers = 2
# d_model = 4
# d_ff = 2
# dropout = 0.1
# a = Transformer(vocab_size, vocab_size, max_len, max_len, num_heads, num_heads, num_layers, num_layers, d_model, d_ff,
#                 dropout, dropout)
# optim = Noam(Adam(alpha=1e-4, beta1=0.9, beta2=0.98, epsilon=1e-9), d_model, 2, 4000)
# a.set_optimizer(optim)
# cri = CrossEntropyLoss(ignore_index=0)
# src = np.array([
#     [2, 3, 4, 5, 0, 0],
#     [4, 7, 2, 1, 8, 0]
# ])
# src_mask = (src != 0).astype(int)[:, None, :]
# tgt = np.array([
#     [7, 2, 6, 1, 1, 1, 0],
#     [2, 3, 2, 2, 0, 0, 0]
# ])
# t = np.array([1, 2, 3, 4, 5, 6, 0, 0, 8, 0, 0, 0, 3, 3])
# tgt_mask = np.triu(np.ones((7, 7)), k=1).astype(int)
# tgt_mask = np.logical_not(tgt_mask)
# tgt_mask2 = (tgt != 0).astype(int)[:, None, :]
# tgt_mask = tgt_mask & tgt_mask2
# o, _ = a.forward(src, tgt, src_mask, tgt_mask, True)
# print(o.shape)
# loss = cri.forward(o.reshape(-1, vocab_size), t)
# print(loss.sum())
# grad = cri.backward()
# print(grad.shape)
# a.backward(grad.reshape(2, 7, 10))
# print(a.encoder.input_embedding.w, end='\n\n')
# print(a.encoder.input_embedding.grad_w, end='\n\n')
# a.update_weights()
# print(a.encoder.input_embedding.w, end='\n\n')
# for i in range(500):
#     o, _ = a.forward(src, tgt, src_mask, tgt_mask, True)
#     loss = cri.forward(o.reshape(-1, vocab_size), t)
#     print(loss.sum())
#     grad = cri.backward()
#     a.backward(grad.reshape(2, 7, 10))
#     a.update_weights()
