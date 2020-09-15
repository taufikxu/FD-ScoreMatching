import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn
import numpy as np

from Tools import FLAGS, load_config
from Tools.logger import save_context, Logger, CheckpointIO
from library import inputs, trainer_baseline, trainer_fd
from library.utils import get_ydist, get_zdist, update_average, Evaluator

# from gan_training import utils
# from gan_training.train import Trainer, update_average
# from gan_training.logger import Logger
# from gan_training.checkpoints import CheckpointIO
# from gan_training.inputs import get_dataset
# from gan_training.distributions import get_ydist, get_zdist
# from gan_training.eval import Evaluator
# from gan_training.config import (
#     load_config,
#     build_models,
#     build_optimizers,
#     build_lr_scheduler,
# )

KEY_ARGUMENTS = load_config(FLAGS.config_file)
KEY_ARGUMENTS = ["trainer.name"] + KEY_ARGUMENTS
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)


torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

nlabels = FLAGS.y_dist.dim
batch_size = FLAGS.training.batch_size

checkpoint_io = CheckpointIO(checkpoint_dir=MODELS_FOLDER)
logger = Logger(log_dir=SUMMARIES_FOLDER)

itr = inputs.get_data_iter(batch_size)
GNet, GOptim = inputs.get_generator_optimizer()
DNet, DOptim = inputs.get_discriminator_optimizer()
GNet_test = copy.deepcopy(GNet)
update_average(GNet_test, GNet, 0.0)
ydist = get_ydist(**vars(FLAGS.y_dist))
zdist = get_zdist(**vars(FLAGS.z_dist))

checkpoint_io.register_modules(GNet=GNet, GOptim=GOptim, DNet=DNet, DOptim=DOptim)
checkpoint_io.register_modules(GNet_test=GNet_test)

trainer_dict = {"baseline": trainer_baseline, "fd": trainer_fd}
trainer_used = trainer_dict[FLAGS.trainer.name]
trainer = trainer_used.Trainer(GNet, DNet, GOptim, DOptim, **vars(FLAGS.trainer.kwargs))
evaluator = Evaluator(GNet_test, zdist, ydist,)


# Distributions


# Train
tstart = t0 = time.time()
inception_every = FLAGS.training.inception_every
# Training loop
print("Start training...")
for it in range(FLAGS.training.n_iter):

    x_real, y = itr.__next__()
    it += 1

    d_lr = DOptim.param_groups[0]["lr"]
    g_lr = GOptim.param_groups[0]["lr"]
    logger.add("learning_rates", "discriminator", d_lr, it=it)
    logger.add("learning_rates", "generator", g_lr, it=it)

    x_real, y = x_real.to(device), y.to(device)
    y.clamp_(None, nlabels - 1)

    # Discriminator updates
    z = zdist.sample((batch_size,))
    dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
    logger.add("losses", "discriminator", dloss, it=it)
    logger.add("losses", "regularizer", reg, it=it)

    # Generators updates
    if ((it + 1) % FLAGS.training.d_steps) == 0:
        z = zdist.sample((batch_size,))
        gloss = trainer.generator_trainstep(y, z)
        logger.add("losses", "generator", gloss, it=it)

        update_average(
            GNet_test, GNet, beta=FLAGS.training.model_average_beta,
        )

    if it % 50 == 0:
        logger.log_info(it + 1, text_logger.info, ["losses", "learning_rates"])

    # (i) Sample if necessary
    if (it % FLAGS.training.sample_every) == 0:
        print("Creating samples...")
        x = evaluator.create_samples()
        logger.add_imgs(x, it)

    # (ii) Compute inception if necessary
    if inception_every > 0 and ((it + 1) % inception_every) == 0:
        inception_mean, inception_std = evaluator.compute_inception_score()
        logger.add("inception_score", "mean", inception_mean, it=it)
        logger.add("inception_score", "stddev", inception_std, it=it)
        logger.log_info(it + 1, text_logger.info, ["inception_score"])

    # (iii) Backup if necessary
    if ((it + 1) % FLAGS.training.backup_every) == 0:
        print("Saving backup...")
        checkpoint_io.save("model_%08d.pt" % it)
        logger.save()
