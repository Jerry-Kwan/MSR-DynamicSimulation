from my_transformer import Adam, Noam, CrossEntropyLoss, PositionalEncoding, Embedding, Linear, PositionwiseFFN, MultiHeadAttention, LayerNorm, EncoderLayer, DecoderLayer, Encoder, Decoder
import numpy as np
import torch

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

a = Decoder(5, 2, 2, 3, 2, 0.1, 10)
tgt = np.array([
    [4, 2, 3, 0, 1],
    [3, 4, 1, 0, 2]
])
print(a.forward(tgt, np.ones((2, 5, 5)), np.random.randn(2, 3, 3), np.ones((2, 1, 3)), True))
print(a.backward(np.ones((2, 5, 5))))
