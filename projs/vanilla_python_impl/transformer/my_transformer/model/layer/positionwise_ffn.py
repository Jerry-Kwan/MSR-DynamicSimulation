from .activation import ReLU
from .linear import Linear
from .dropout import Dropout


class PositionwiseFFN:
    """
    Implementation of Position-Wise Feed-Forward Network.
    """

    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1):
        self.fc_1 = Linear(in_features=d_model, out_features=d_ff)
        self.activation = ReLU()
        self.fc_2 = Linear(in_features=d_ff, out_features=d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, X, training=True):
        """
        Reference:
            https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html
        """
        X = self.fc_1.forward(X, training)
        X = self.activation.forward(X)
        X = self.fc_2.forward(X, training)
        X = self.dropout.forward(X, training)

        return X

    def backward(self, grad):
        grad = self.dropout.backward(grad)
        grad = self.fc_2.backward(grad)
        grad = self.activation.backward(grad)
        grad = self.fc_1.backward(grad)

        return grad

    def set_optimizer(self, optimizer):
        self.fc_1.set_optimizer(optimizer)
        self.fc_2.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.fc_1.update_weights(layer_num)
        layer_num = self.fc_2.update_weights(layer_num)

        return layer_num
