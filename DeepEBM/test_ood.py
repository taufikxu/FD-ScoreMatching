import glob
import os
import pickle

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# import torchvision
from matplotlib import pyplot as plt

import ESM
from ESM import inputs

# from ESM.sampling_mdsm import Annealed_Langevin_E, SS_denoise

from Utils import config, flags

from sklearn import metrics


matplotlib.use("Agg")
FLAGS = flags.FLAGS
config.load_config(FLAGS.config_file)
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models = glob.glob(FLAGS.old_model)
pmodel_dat = dict({})
for mid, model in enumerate(models):
    # model = FLAGS.old_model
    dirname = os.path.dirname(model)
    basename = os.path.basename(model)
    config_path = os.path.join(dirname, "..", "source", "configs_dict.pkl")
    summary_path = os.path.join(dirname, "..", "summary")
    with open(config_path, "rb") as f:
        new_dict = pickle.load(f)
    if "dilation" not in new_dict:
        new_dict["dilation"] = False
    FLAGS.set_dict(new_dict)

    torch.manual_seed(1234)
    np.random.seed(1235)

    M = 2
    svhn_loader = inputs.iter_func["svhn"](M, train=False)
    cifar_loader = inputs.iter_func["cifar"](M, train=False)
    imagenet_loader = inputs.iter_func["imagenet"](M, train=False)

    netE = inputs.get_model()
    train_loader = inputs.iter_func[FLAGS.dataset](100, train=True)

    netE = netE.to(device)
    hw, in_chan = netE.hw, netE.inchan
    netE = nn.DataParallel(netE)

    netE.load_state_dict(torch.load(model))
    svhn_list, cifar_list, imagenet_list = [], [], []

    with torch.no_grad():
        score_list = []
        for tid in range(10):
            dat = train_loader.__next__()
            dat = dat.to(device)
            dat.requires_grad_(True)
            score = netE(dat).mean()
            score_list.append(score.item())
        score_list = np.mean(score_list)

    for tid in range(1000):
        dat = svhn_loader.__next__()
        dat = F.interpolate(dat, size=(hw, hw), align_corners=False, mode="bilinear")
        dat = dat.to(device)
        # dat.requires_grad_(True)
        score = netE(dat).mean()
        svhn_list.append(np.abs(score.item() - score_list))

        dat = cifar_loader.__next__()
        dat = F.interpolate(dat, size=(hw, hw), align_corners=False, mode="bilinear")
        dat = dat.to(device)
        # dat.requires_grad_(True)
        score = netE(dat).mean()
        cifar_list.append(np.abs(score.item() - score_list))

        dat = imagenet_loader.__next__()
        dat = F.interpolate(dat, size=(hw, hw), align_corners=False, mode="bilinear")
        dat = dat.to(device)
        # dat.requires_grad_(True)
        score = netE(dat).mean()
        imagenet_list.append(np.abs(score.item() - score_list))

    svhn_list = np.array(svhn_list).reshape(-1)
    cifar_list = np.array(cifar_list).reshape(-1)
    imagenet_list = np.array(imagenet_list).reshape(-1)

    minv = 0
    maxv = min(max(np.max(svhn_list), np.max(cifar_list), np.max(imagenet_list)), 3000)
    bins = np.linspace(minv, maxv, 1000)

    figure = plt.figure()
    plt.hist(svhn_list, bins, alpha=0.5, label="SVHN")
    plt.hist(cifar_list, bins, alpha=0.5, label="CIFAR10")
    plt.hist(imagenet_list, bins, alpha=0.5, label="ImageNet")
    plt.legend()
    figure.savefig(
        os.path.join(
            summary_path, os.path.splitext(basename)[0] + "OOD_test_{}.pdf".format(M)
        )
    )

    if FLAGS.dataset.lower() == "svhn":
        gt_list = []
        for _ in range(1000):
            dat = svhn_loader.__next__()
            dat = F.interpolate(dat, size=(hw, hw), align_corners=False, mode="bilinear")
            dat = dat.to(device)
            # dat.requires_grad_(True)
            score = netE(dat).mean()
            gt_list.append(np.abs(score.item() - score_list))
        gt_list = np.array(gt_list)
    elif FLAGS.dataset.lower() == "cifar":
        gt_list = []
        for _ in range(1000):
            dat = cifar_loader.__next__()
            dat = F.interpolate(dat, size=(hw, hw), align_corners=False, mode="bilinear")
            dat = dat.to(device)
            # dat.requires_grad_(True)
            score = netE(dat).mean()
            gt_list.append(np.abs(score.item() - score_list))
        gt_list = np.array(gt_list)
    elif FLAGS.dataset.lower() == "imagenet":
        gt_list = []
        for _ in range(1000):
            dat = imagenet_loader.__next__()
            dat = F.interpolate(dat, size=(hw, hw), align_corners=False, mode="bilinear")
            dat = dat.to(device)
            # dat.requires_grad_(True)
            score = netE(dat).mean()
            gt_list.append(np.abs(score.item() - score_list))
        gt_list = np.array(gt_list)

    dataset_names = ["svhn", "cifar", "imagenet"]
    for did, target_list in enumerate([svhn_list, cifar_list, imagenet_list]):
        pred_list = []
        for i in range(1000):
            pred_list.append([1, gt_list[i]])
        for i in range(1000):
            pred_list.append([2, target_list[i]])
        pred_list.sort(key=lambda x: x[1])

        y = np.array([x[0] for x in pred_list])
        pred = np.array([x[1] for x in pred_list])
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
        print(model, dataset_names[did], M, metrics.auc(fpr, tpr))
    with open(model + ".pkl", "wb") as f:
        pickle.dump("loaded", f)
