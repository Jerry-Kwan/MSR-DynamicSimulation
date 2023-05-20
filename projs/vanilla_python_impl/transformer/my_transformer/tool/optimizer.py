try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class Optimizer:
    """Basic class of Optimizer."""

    def __init__(self):
        pass

    def update(self):
        raise NotImplementedError


class Scheduler:
    """Basic class of Scheduler."""

    def __init__(self):
        pass

    def update(self):
        raise NotImplementedError


class Adam(Optimizer):
    """
    Adam Optimizer.

    Reference:
        https://blog.csdn.net/m0_37944102/article/details/104340723
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        # gradient, weights, v, m, v_hat, m_hat, share the same shape
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(self.beta1, t))
        v_hat = v / (1 - np.power(self.beta2, t))

        weights -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights, v, m, v_hat, m_hat


class Noam(Scheduler):
    """
    Noam Scheduler.

    lr = scale_factor * ((d_model ** (-0.5)) * adj_step)
    adj_step = min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    """

    def __init__(self, optimizer, d_model, scale_factor=1, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.steps_num = 0

    @staticmethod
    def compute_learning_rate(scale_factor, d_model, steps_num, warmup_steps):
        return scale_factor * (d_model**(-0.5) * min(steps_num**(-0.5), steps_num * warmup_steps**(-1.5)))

    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        self.steps_num += 1
        self.optimizer.alpha = self.compute_learning_rate(self.scale_factor, self.d_model, self.steps_num,
                                                          self.warmup_steps)
        return self.optimizer.update(gradient, weights, v, m, v_hat, m_hat, t)
