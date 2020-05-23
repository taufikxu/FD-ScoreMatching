import glob
import os
import pickle

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from matplotlib import pyplot as plt

import ESM
from ESM import inputs
from ESM.sampling_mdsm import Annealed_Langevin_E, SS_denoise

import Utils
from Utils import config, flags

matplotlib.use("Agg")
# FILES_TO_BE_SAVED = ["./", "./configs", "./ESM"]
# KEY_ARGUMENTS = ["data"]
# CONFIG = {"FILES_TO_BE_SAVED": FILES_TO_BE_SAVED, "KEY_ARGUMENTS": KEY_ARGUMENTS}
FLAGS = flags.FLAGS
config.load_config(FLAGS.config_file)
if FLAGS.gpu.lower() not in ["-1", "none", Utils.config.notValid.lower()]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

torch.manual_seed(1234)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(1235)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    all_models = glob.glob(FLAGS.old_model)
    torch.cuda.manual_seed(1234)
    print(all_models)
    for model in all_models:
        dirname = os.path.dirname(model)
        basename = os.path.basename(model)
        config_path = os.path.join(dirname, "..", "source", "configs_dict.pkl")
        summary_path = os.path.join(dirname, "..", "summary")
        with open(config_path, "rb") as f:
            new_dict = pickle.load(f)
        if "dilation" not in new_dict:
            new_dict["dilation"] = False
        FLAGS.set_dict(new_dict)
        # FLAGS.batch_size = 32
        # print(FLAGS.dataset)

        netE = inputs.get_model()
        netE = netE.to(device)
        hw, inchan = netE.hw, netE.inchan
        netE = nn.DataParallel(netE)

        if FLAGS.annealing_schedule == "exp":
            Nsampling = 2000
            Tmax, Tmin = FLAGS.Tmax, FLAGS.Tmin
            T = Tmax * np.exp(
                -np.linspace(0, Nsampling - 1, Nsampling)
                * (np.log(Tmax / Tmin) / Nsampling)
            )
            T = np.concatenate((Tmax * np.ones((500,)), T), axis=0)
            T = np.concatenate((T, Tmin * np.linspace(1, 0, 200)), axis=0)

        elif FLAGS.annealing_schedule == "lin":
            Nsampling = 2000
            Tmax, Tmin = FLAGS.Tmax, FLAGS.Tmin
            T = np.linspace(Tmax, Tmin, Nsampling)
            T = np.concatenate((Tmax * np.ones((500,)), T), axis=0)
            T = np.concatenate((T, Tmin * np.linspace(1, 0, 200)), axis=0)

        netE.load_state_dict(torch.load(model))

        if FLAGS.sample_mode == "visualize":
            n_batches = 1
            FLAGS.batch_size = FLAGS.n_samples_save
        elif FLAGS.sample_mode == "OOD":
            continue
        else:
            n_batches = int(np.ceil(FLAGS.n_samples_save / FLAGS.batch_size))

        denoise_samples = []
        print("sampling starts")
        for i in range(n_batches):
            initial_x = 0.5 + torch.randn(FLAGS.batch_size, inchan, hw, hw).to(device)
            x_list, E_trace = Annealed_Langevin_E(
                netE, initial_x, FLAGS.sample_step_size, T, 100
            )

            x_denoise = x_list[-1][:].to(device)
            for _ in range(1):
                x_denoise = SS_denoise(x_denoise, netE, 0.1)
            denoise_samples.append(x_denoise)
            print("batch {}/{} finished".format((i + 1), n_batches))

        denoise_samples = torch.cat(denoise_samples, 0)
        imgs = torchvision.utils.make_grid(denoise_samples, nrow=10)
        torchvision.utils.save_image(imgs, os.path.splitext(model)[0] + ".jpg", nrow=8)


if __name__ == "__main__":
    main()
