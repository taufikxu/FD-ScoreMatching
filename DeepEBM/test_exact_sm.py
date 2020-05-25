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
FLAGS = flags.FLAGS
config.load_config(FLAGS.config_file)
if FLAGS.gpu.lower() not in ["-1", "none", Utils.config.notValid.lower()]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(1235)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    results_dict = dict({})
    all_models = glob.glob(FLAGS.old_model)
    torch.cuda.manual_seed(1234)
    print(all_models)
    for model in all_models:
        dirname = os.path.dirname(model)
        config_path = os.path.join(dirname, "..", "source", "configs_dict.pkl")
        with open(config_path, "rb") as f:
            new_dict = pickle.load(f)
        if "dilation" not in new_dict:
            new_dict["dilation"] = False
        FLAGS.set_dict(new_dict)

        netE = inputs.get_model()
        netE = netE.to(device)
        hw, inchan = netE.hw, netE.inchan
        netE = nn.DataParallel(netE)
        netE.load_state_dict(torch.load(model))

        test_loader = ESM.inputs.iter_func[FLAGS.dataset](
            100, train=False, infinity=False
        )
        loss_list = []
        for i, dat in enumerate(test_loader):
            dat = dat.to(device)
            tloss = ESM.loss_train.exact_score_matching(netE, dat, hw * hw * inchan)
            loss_list.append(tloss.item())
            if i == 14:
                break
        print(model, np.mean(loss_list))
        results_dict[model] = loss_list
    for m in results_dict:
        print(m, np.mean(results_dict[m]))
    with open("tmp.pkl", "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    main()
