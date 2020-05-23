import torch
from torch import nn

import ESM.data as dataset_iters
import ESM.models as models
import ESM.loss_train as losses
from Utils import flags

FLAGS = flags.FLAGS
iter_func = {
    "cifar": dataset_iters.inf_train_gen_cifar,
    "svhn": dataset_iters.inf_train_gen_svhn,
    "mnist": dataset_iters.inf_train_gen_mnist,
    "celeba": dataset_iters.inf_train_gen_celeba,
    "celeba32": dataset_iters.inf_train_gen_celeba32,
    "imagenet": dataset_iters.inf_train_gen_imagenet,
    "fashionmnist": dataset_iters.inf_train_gen_fashionmnist,
}
model_dict = {
    "cifar_resnet18": models.Res18_Quadratic,
    "svhn_resnet18": models.Res18_Quadratic,
    "celeba_resnet18": models.Res18_Quadratic,
    "celeba32_resnet18": models.Res18_Quadratic,
    "mnist_resnet18": models.Res18_Quadratic,
    "mnist_resnet6": models.Res6_Quadratic,
    "mnist_mlp": models.MLP_Quadratic,
    "fashionmnist_resnet18": models.Res18_Quadratic,
    "cifar_resnet18_dense": models.Res18_Quadratic_dense,
    "cifar_resnet18_unet": models.Res18_Quadratic_unet,
    "imagenet_resnet34": models.Res34_Quadratic_Imagenet,
}
hw_dict = {
    "cifar": (32, 3),
    "svhn": (32, 3),
    "mnist": (32, 1),
    "fashionmnist": (32, 1),
    "imagenet": (224, 3),
    "celeba": (64, 3),
    "celeba32": (32, 3),
}


def get_data_iter():
    return iter_func[FLAGS.dataset.lower()](FLAGS.batch_size)


def get_model():
    if FLAGS.activation.lower() == "elu":
        activation = nn.ELU()
    elif FLAGS.activation.lower() == "relu":
        activation = nn.ReLU()
    elif FLAGS.activation.lower() == "softplus":
        activation = nn.Softplus()
    elif FLAGS.activation.lower() == "nsoftplus":
        activation = models.NormalizedSoftplus()
    else:
        raise ValueError("Unknown Activation.")

    if FLAGS.normalization.lower() in ["none", "false"]:
        normalization = False
    elif FLAGS.normalization.lower() in ["t", "true", "1"]:
        normalization = True
    else:
        normalization = FLAGS.normalization

    name = FLAGS.dataset.lower() + "_" + FLAGS.model_name
    hw = hw_dict[FLAGS.dataset.lower()]
    model = model_dict[name](
        hw[1], FLAGS.n_chan, hw[0], normalize=normalization, AF=activation
    )
    return model


def get_optimizer_scheduler(model):
    # setup optimizer and lr scheduler
    if FLAGS.optimizer.lower() == "adam":
        params = {"lr": FLAGS.max_lr, "betas": (FLAGS.beta1, FLAGS.beta2)}
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif FLAGS.optimizer.lower() == "sgd":
        params = {"lr": FLAGS.max_lr, "momentum": FLAGS.beta1}
        optimizer = torch.optim.SGD(model.parameters(), **params)

    if FLAGS.lr_schedule == "exp":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(FLAGS.n_iter / 6))

    elif FLAGS.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, FLAGS.n_iter, eta_min=FLAGS.min_lr, last_epoch=-1
        )
    return optimizer, scheduler


def get_train_loss():
    hw = hw_dict[FLAGS.dataset]
    dim = hw[0] * hw[0] * hw[1]
    if FLAGS.loss_func.lower() == "mdsm_baseline":
        loss_func = losses.mdsm_baseline
    elif FLAGS.loss_func.lower() == "mdsm_fd":
        loss_func = losses.mdsm_fd
    elif FLAGS.loss_func.lower() == "mdsm_fd_abs":
        loss_func = losses.mdsm_fd_abs
    elif FLAGS.loss_func.lower() == "mdsm_fd_mutv":
        loss_func = losses.mdsm_fd_mutv
    elif FLAGS.loss_func.lower() == "mdsm_tracetrick":
        loss_func = losses.mdsm_tracetrick
    elif FLAGS.loss_func.lower() == "mdsm_ssm_fd":
        loss_func = losses.mdsm_ssm_fd
    elif FLAGS.loss_func.lower() == "mdsm_fd_nop":
        loss_func = losses.mdsm_fd_nop
    elif FLAGS.loss_func.lower() == "ssm":
        loss_func = losses.ssm
    elif FLAGS.loss_func.lower() == "ssm_vr":
        loss_func = losses.ssm_vr
    elif FLAGS.loss_func.lower() == "ssm_fd":
        loss_func = losses.ssm_fd
    elif FLAGS.loss_func.lower() == "ssm_fd_nop":
        loss_func = losses.ssm_fd_nop

    if "mdsm" in FLAGS.loss_func.lower():

        def myloss(energy, sample, sigmas, sigma02):
            return loss_func(energy, sample, sigmas, sigma02, dim)

    else:
        myloss = loss_func

    return myloss


def get_eval_loss():
    if "mdsm" in FLAGS.loss_func.lower():

        hw = hw_dict[FLAGS.dataset]
        dim = hw[0] * hw[0] * hw[1]

        def myloss(energy, sample, sigmas, sigma02):
            return losses.mdsm_baseline(energy, sample, sigmas, sigma02, dim)

    else:
        myloss = losses.ssm_vr

    return myloss
