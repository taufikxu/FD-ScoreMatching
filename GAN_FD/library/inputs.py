import torch
from torch import nn

import library.dataset_iters as dataset_iters
import library.models as models
from Tools import FLAGS

hw_dict = {
    "cifar10": (32, 3),
    "svhn": (32, 3),
    "mnist": (32, 1),
    "fashionmnist": (32, 1),
    "imagenet": (128, 3),
    "celeba": (64, 3),
    "celeba32": (32, 3),
}


def get_data_iter(batch_size=None, train=True, infinity=True, subset=0):
    if batch_size is None:
        batch_size = FLAGS.batch_size
    return dataset_iters.inf_train_gen(batch_size, train, infinity, subset)


def get_optimizer(params, opt_name, lr, beta1, beta2):
    if opt_name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2))
    elif opt_name.lower() == "nesterov":
        optim = torch.optim.SGD(
            params, lr, momentum=beta1, weight_decay=FLAGS.c_weight_decay, nesterov=True
        )
    return optim


def get_generator_optimizer():
    module = models.G_dict[FLAGS.generator.name.lower()]
    hw, c = hw_dict[FLAGS.dataset.lower()]
    G = module(**vars(FLAGS.generator.kwargs)).to(FLAGS.device)

    optim_kwargs = vars(FLAGS.generator.optim)
    optim_kwargs.update({"params": G.parameters()})
    optim = get_optimizer(**optim_kwargs)
    return G, optim


def get_discriminator_optimizer():
    module = models.D_dict[FLAGS.discriminator.name.lower()]
    hw, c = hw_dict[FLAGS.dataset.lower()]
    D = module(**vars(FLAGS.discriminator.kwargs)).to(FLAGS.device)

    optim_kwargs = vars(FLAGS.discriminator.optim)
    optim_kwargs.update({"params": D.parameters()})
    optim = get_optimizer(**optim_kwargs)
    return D, optim

