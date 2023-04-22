import numpy as np


class ReLU():
    def forward(self, x):
        self.old_x = np.copy(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        return np.where(self.old_x > 0, grad, 0)


class Sigmoid():
    def forward(self, x):
        self.old_y = np.exp(x) / (1. + np.exp(x))
        return self.old_y

    def backward(self, grad):
        return self.old_y * (1. - self.old_y) * grad


class Softmax():
    def forward(self, x):
        # self.old_y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        # handle overflow
        t = np.exp(x - x.max(axis=1, keepdims=True))
        self.old_y = t / t.sum(axis=1, keepdims=True)
        return self.old_y

    def backward(self, grad):
        return self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, None])


class CrossEntropy():
    def forward(self, x, y):
        self.old_x = x.clip(min=1e-8, max=None)
        self.old_y = y
        return (np.where(y == 1, -np.log(self.old_x), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.old_y == 1, -1 / self.old_x, 0)


class Linear():
    def __init__(self, n_in, n_out):
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.biases = np.zeros(n_out)

    def forward(self, x):
        self.old_x = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (np.matmul(self.old_x[:, :, None], grad[:, None, :])).mean(axis=0)
        return np.dot(grad, self.weights.transpose())


class Model():
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def compute_loss(self, x, y):
        # x should be the output of the forward method
        return self.cost.forward(x, y)

    def backward(self):
        grad = self.cost.backward()

        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)
