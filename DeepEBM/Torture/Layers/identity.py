import torch.nn as nn


class IdentityLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x
