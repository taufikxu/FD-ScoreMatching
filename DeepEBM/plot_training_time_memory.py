import os
import shutil
import time
import copy
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision

from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from Torture.Models import get_model
from Torture.shortcuts import build_optimizers
from Torture.utils import distributions
from Torture import shortcuts

from ESM.models import (
    Res18_Quadratic_MNIST,
    Res12_Quadratic_MNIST,
    MLP_Quadratic_MNIST,
    NormalizedSoftplus,
)
from ESM.data import inf_train_gen_cifar, inf_train_gen_mnist
from ESM.loss import loss_dict, score_matching

FILES_TO_BE_SAVED = ["./", "./configs", "./ESM"]
KEY_ARGUMENTS = ["data"]
CONFIG = {"FILES_TO_BE_SAVED": FILES_TO_BE_SAVED, "KEY_ARGUMENTS": KEY_ARGUMENTS}

FLAGS = flags.FLAGS
KEY_ARGUMENTS += config.load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, CONFIG)
shutil.copy(FLAGS.config_file, os.path.join(SUMMARIES_FOLDER, "config.yaml"))
print = text_logger.info

torch.manual_seed(1234)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(12345)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(FLAGS.rand_seed)

logger = Logger(log_dir=SUMMARIES_FOLDER)

AF = nn.ELU()
itr = inf_train_gen_mnist(FLAGS.batch_size)
if FLAGS.model.lower() in ["cnn", "resnet"]:
    netE = Res18_Quadratic_MNIST(3, FLAGS.n_chan, 32, normalize=False, AF=AF)
else:
    netE = MLP_Quadratic_MNIST(1, FLAGS.n_chan, 32, normalize=False, AF=AF)
print(str(netE))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netE = netE.to(device)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netE=netE)

# setup optimizer and lr scheduler
params = {"lr": FLAGS.max_lr, "betas": (0.9, 0.999)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizerE, FLAGS.n_iter, eta_min=0, last_epoch=-1
)

# train
print_interval = 50
max_iter = FLAGS.n_iter + FLAGS.net_indx
batchSize = FLAGS.batch_size
sigma0 = FLAGS.sigma0
sigma02 = sigma0 ** 2

sigmas_np = np.linspace(FLAGS.min_noise, FLAGS.max_noise, batchSize)
sigmas = torch.Tensor(sigmas_np).view(-1, 1, 1, 1).to(device)
start = end = time.time()

if FLAGS.loss.find("dsm") != -1:
    loss_func_ = loss_dict[FLAGS.loss]

    def loss_func(netE, x_real):
        return loss_func_(netE, x_real, sigmas, sigma02)


else:
    loss_func_ = loss_dict[FLAGS.loss]

    def loss_func(netE, x_real):
        return loss_func_(netE, x_real)


training_time, training_loss = [], []
eval_loss = []
time_buffer = 0
loss_buffer = [0] * 100
for i in range(FLAGS.n_iter):
    x_real = itr.__next__().to(device)
    x_real += torch.randn_like(x_real) * 0.001
    start = time.time()

    loss = loss_func(netE, x_real)
    optimizerE.zero_grad()
    loss.backward()
    time_dur = time.time() - start
    time_buffer += time_dur
    torch.nn.utils.clip_grad_norm_(netE.parameters(), 0.1)
    optimizerE.step()

    training_loss.append(loss.item())
    scheduler.step()
    if i % 100 == 99:
        eval_loss.append(score_matching(netE, x_real))
        training_time.append(time_buffer)
        time_buffer = 0.0
        info = (i, training_time[-1], training_loss[-1])
        print(str(info))

training_time = np.array(training_time)
training_loss = np.array(training_loss)
eval_loss = np.array(eval_loss)
np.savetxt(os.path.join(SUMMARIES_FOLDER, "training_time.txt"), training_time)
np.savetxt(os.path.join(SUMMARIES_FOLDER, "training_loss.txt"), training_loss)
np.savetxt(os.path.join(SUMMARIES_FOLDER, "eval_loss.txt"), eval_loss)
