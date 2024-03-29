# import os
import time

import torch
import torch.nn as nn
import numpy as np

from ESM import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
FILES_TO_BE_SAVED = ["./", "./configs", "./ESM"]
CONFIG = {"FILES_TO_BE_SAVED": FILES_TO_BE_SAVED, "KEY_ARGUMENTS": KEY_ARGUMENTS}
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, CONFIG)

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(1235)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(1236)

logger = Logger(log_dir=SUMMARIES_FOLDER)

itr = inputs.get_data_iter()
netE = inputs.get_model()
netE = netE.to(device)
if FLAGS.n_gpus > 1:
    netE = nn.DataParallel(netE)

optimizerE, scheduler = inputs.get_optimizer_scheduler(netE)
loss_func = inputs.get_train_loss()
loss_eval = inputs.get_eval_loss()

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netE=netE)

# train
print_interval = 50
max_iter = FLAGS.n_iter + FLAGS.net_indx
batchSize = FLAGS.batch_size
sigma0 = FLAGS.sigma0
sigma02 = sigma0 ** 2

if FLAGS.noise_distribution == "exp":
    sigmas_np = np.logspace(
        np.log10(FLAGS.min_noise), np.log10(FLAGS.max_noise), batchSize
    )
elif FLAGS.noise_distribution == "lin":
    sigmas_np = np.linspace(FLAGS.min_noise, FLAGS.max_noise, batchSize)

sigmas = torch.Tensor(sigmas_np).view(-1, 1).to(device)
time_dur = 0.0
netE.train()
for i in range(FLAGS.net_indx, FLAGS.net_indx + FLAGS.n_iter):
    x_real = itr.__next__().to(device)
    x_real = x_real + torch.randn_like(x_real) * FLAGS.ssm_eps
    start_time = time.time()
    tloss = loss_func(netE, x_real)
    optimizerE.zero_grad()
    tloss.backward()
    torch.nn.utils.clip_grad_norm_(netE.parameters(), FLAGS.clip_value)
    optimizerE.step()
    time_dur += time.time() - start_time
    scheduler.step()

    if i % print_interval == 0:
        netE.eval()
        E_real = netE(x_real).mean()
        E_noise = netE(torch.rand_like(x_real)).mean()
        ntloss = loss_eval(netE, x_real)
        netE.train()

        str_meg = "Iteration {}/{} ({:.0f}%), E_real {:e},"
        str_meg += " E_noise {:e}, tLoss {:e}, Normalized Loss {:e}, time {:4.3f}"
        text_logger.info(
            str_meg.format(
                i,
                max_iter,
                100 * ((i) / max_iter),
                E_real.item(),
                E_noise.item(),
                tloss.item(),
                ntloss.item(),
                time_dur,
            )
        )
        time_dur = 0.0

        logger.add("training", "E_real", E_real.item(), i + 1)
        logger.add("training", "E_noise", E_noise.item(), i + 1)
        logger.add("training", "loss", tloss.item(), i + 1)
        del E_real
        del E_noise
        del ntloss
        del tloss

    if (i + 1) % FLAGS.save_every == 0:
        text_logger.info("-" * 50)
        file_name = "model" + str(i + 1) + ".pt"
        torch.save(netE.state_dict(), MODELS_FOLDER + "/" + file_name)
