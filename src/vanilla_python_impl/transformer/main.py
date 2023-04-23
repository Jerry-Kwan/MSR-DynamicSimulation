# parser: dtype
from my_transformer import Adam, Noam, CrossEntropyLoss, PositionalEncoding, Embedding, Linear, PositionwiseFFN
import numpy as np
import torch

# a = CrossEntropyLoss()
# x_input = torch.randn(4, 3)  #随机生成输入
# y_target = torch.tensor([1, 2, 0, 1])  #设置输出具体值 print('y_target\n',y_target)
# print(np.mean(a.forward(x_input, y_target)))
# print(a.backward())
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
