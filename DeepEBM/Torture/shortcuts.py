import torch
import numpy as np
from torch import optim
import torch.nn.init as init
from torch import nn
from Utils.flags import FLAGS


# Function for Initialization
def weight_init(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta=0.999):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def get_nsamples(data_loader, N):
    x = []
    n = 0
    while n < N:
        dat = next(iter(data_loader))
        if isinstance(dat, tuple) is True or isinstance(dat, list) is True:
            x_tmp = dat[0]
        else:
            x_tmp = dat
        x.append(x_tmp)
        n += x_tmp.size(0)
    x = torch.cat(x, dim=0)[:N]
    return x


def build_optimizers(parameters, optimizer, lr, prefix=None):
    kwargs = dict({})
    if prefix is not None:
        length = len(prefix)
        args = FLAGS.get_dict()
        for k in args:
            if k.find(prefix) == 0:
                kwargs[k[length:]] = args[k]
    # Optimizers
    if optimizer == 'rmsprop':
        opt = optim.RMSprop(parameters, lr=lr, **kwargs)
    elif optimizer == 'adam':
        opt = optim.Adam(parameters, lr=lr, **kwargs)
    elif optimizer == 'sgd':
        opt = optim.SGD(parameters, lr=lr, **kwargs)
    return opt
