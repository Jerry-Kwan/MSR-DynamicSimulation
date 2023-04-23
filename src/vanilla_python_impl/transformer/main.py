# parser: dtype
from my_transformer import Adam, Noam, CrossEntropyLoss
import numpy as np
import torch

a = CrossEntropyLoss()
x_input = torch.randn(4, 3)  #随机生成输入
y_target = torch.tensor([1, 2, 0, 1])  #设置输出具体值 print('y_target\n',y_target)
print(np.mean(a.forward(x_input, y_target)))
print(a.backward())
x_input.requires_grad_()
crossentropyloss = torch.nn.CrossEntropyLoss()
crossentropyloss_output = crossentropyloss(x_input, y_target)
print('crossentropyloss_output:\n', crossentropyloss_output)
crossentropyloss_output.backward()
print(x_input.grad * 4)