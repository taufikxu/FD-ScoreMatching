import os
import shutil
import time
import copy

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

from ESM.models import Res18_Quadratic
from ESM.data import inf_train_gen_cifar

FILES_TO_BE_SAVED = ["./", './configs', './ESM']
KEY_ARGUMENTS = ['data', 'esm_eps', 'esm_type']
CONFIG = {
    "FILES_TO_BE_SAVED": FILES_TO_BE_SAVED,
    "KEY_ARGUMENTS": KEY_ARGUMENTS
}

FLAGS = flags.FLAGS
config.load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(
    __file__, FLAGS, CONFIG)
shutil.copy(FLAGS.config_file, os.path.join(SUMMARIES_FOLDER, "config.yaml"))
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Res18_Quadratic(3, FLAGS.n_chan, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), 5e-5)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(
    checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(model=model)

logger = Logger(log_dir=SUMMARIES_FOLDER)

torch.cuda.manual_seed(FLAGS.rand_seed)
if FLAGS.dataset == 'cifar':
    itr = inf_train_gen_cifar(FLAGS.batch_size, flip=False)
    netE = Res18_Quadratic(3, FLAGS.n_chan, 32, normalize=False, AF=nn.ELU())
else:
    NotImplementedError('{} unknown dataset'.format(FLAGS.dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netE = netE.to(device)
if FLAGS.n_gpus > 1:
    netE = nn.DataParallel(netE)

# setup optimizer and lr scheduler
params = {'lr': FLAGS.max_lr, 'betas': (0.9, 0.95), 'weight_decay': 1e-5}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
if FLAGS.lr_schedule == 'exp':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizerE,
                                                int(FLAGS.n_iter / 6))

elif FLAGS.lr_schedule == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE,
                                                           FLAGS.n_iter,
                                                           eta_min=1e-6,
                                                           last_epoch=-1)

# train
print_interval = 50
max_iter = FLAGS.n_iter + FLAGS.net_indx
batchSize = FLAGS.batch_size
sigma0 = 0.1
sigma02 = sigma0**2

if FLAGS.noise_distribution == 'exp':
    sigmas_np = np.logspace(np.log10(FLAGS.min_noise),
                            np.log10(FLAGS.max_noise), batchSize)
elif FLAGS.noise_distribution == 'lin':
    sigmas_np = np.linspace(FLAGS.min_noise, FLAGS.max_noise, batchSize)

sigmas = torch.Tensor(sigmas_np).view((batchSize, 1, 1, 1)).to(device)
start_time = time.time()

for i in range(FLAGS.net_indx, FLAGS.net_indx + FLAGS.n_iter):
    x_real = itr.__next__().to(device)
    noise_dsm = torch.randn_like(x_real)
    x_noisy = x_real + sigmas * noise_dsm
    if FLAGS.esm_type == 'sphere':
        v_un = torch.randn_like(x_noisy)
        v_norm = torch.sqrt(torch.sum(v_un**2, dim=(1, 2, 3), keepdim=True))
        v = v_un / v_norm * FLAGS.esm_eps * np.sqrt(32 * 32 * 3)
    elif FLAGS.esm_type == "radermacher":
        v_raw = torch.randn_like(x_noisy)
        v = v_raw.sign() * FLAGS.esm_eps
    elif FLAGS.esm_type == "sphere_add":
        v_direc = noise_dsm * (torch.rand(batchSize, 1, 1, 1).to(device) -
                               0.5).sign()
        v_un = torch.randn_like(x_noisy) * FLAGS.esm_add_pert + v_direc
        v_norm = torch.sqrt(torch.sum(v_un**2, dim=(1, 2, 3), keepdim=True))
        v = v_un / v_norm * FLAGS.esm_eps * np.sqrt(32 * 32 * 3)

    x_noisy = x_noisy.requires_grad_()
    cat_x = torch.cat([x_noisy + v, x_noisy - v], 0)
    logp = -1 * netE(cat_x)
    logp1 = logp[:batchSize]
    logp2 = logp[batchSize:]

    # x_noisy.detach()

    optimizerE.zero_grad()
    first_term = 0.5 * (logp1 - logp2) / sigmas.view(-1)
    second_term = torch.sum(v * (x_noisy - x_real) / sigmas / sigma02,
                            dim=(1, 2, 3))
    # print(first_term.shape, second_term.shape)
    LS_loss = ((first_term + second_term)**2).mean() / FLAGS.esm_eps**2
    LS_loss.backward()
    torch.nn.utils.clip_grad_value_(netE.parameters(), FLAGS.clip_value)
    optimizerE.step()
    scheduler.step()

    if (i + 1) % print_interval == 0:
        time_spent = time.time() - start_time
        start_time = time.time()

        netE.eval()
        x_real = itr.__next__().to(device)
        x_noisy = x_real + sigmas * torch.randn_like(x_real)
        x_noisy = x_noisy.requires_grad_()
        E = -netE(x_noisy).sum()
        score = torch.autograd.grad(E, x_noisy)[0]
        exact_LS_loss = (((score / sigmas +
                           (x_noisy - x_real) / sigmas / sigma02)**2) /
                         batchSize).sum()
        E_real = netE(x_real).mean()
        E_noise = netE(torch.rand_like(x_real)).mean()
        netE.train()

        text_logger.info(
            'Iteration {}/{} ({:.0f}%), E_real {:e}, E_noise {:e}, Normalized Loss {:e}, exact Normalized Loss {:e}, time {:4.1f}'
            .format(i + 1, max_iter, 100 * ((i + 1) / max_iter), E_real.item(),
                    E_noise.item(), (sigma02**2) * (LS_loss.item()),
                    (sigma02**2) * (exact_LS_loss.item()), time_spent))

    if (i + 1) % FLAGS.save_every == 0:
        text_logger.info("-" * 50)
        file_name = "model" + str(i + 1) + '.pt'
        torch.save(netE.state_dict(), MODELS_FOLDER + '/' + file_name)
