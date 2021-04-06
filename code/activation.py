import torch
import torch.nn as nn


class Identity(nn.Module):
        def __init__(self, *args, **kwargs):
                super().__init__()
        def forward(self, x):
                return x


class Activation(nn.Module):

        def __init__(self, name, **params):

                super().__init__()

                if name is None or name == 'identity':
                        self.activation = Identity(**params)
                elif name == 'sigmoid':
                        self.activation = nn.Sigmoid()
                elif name == 'softmax2d':
                        self.activation = nn.Softmax(dim=1, **params)
                elif name == 'softmax':
                        self.activation = nn.Softmax(**params)
                elif name == 'logsoftmax':
                        self.activation = nn.LogSoftmax(**params)
                elif name == 'tanh':
                        self.activation = nn.Tanh()
                elif name == 'argmax':
                        self.activation = ArgMax(**params)
                elif name == 'argmax2d':
                        self.activation = ArgMax(dim=1, **params)
                elif callable(name):
                        self.activation = name(**params)
                else:
                        raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

        def forward(self, x):
                return self.activation(x)