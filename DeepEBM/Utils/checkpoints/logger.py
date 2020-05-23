import logging
import operator
import os
import pickle
import socket
import time

import coloredlogs
import torch
import torchvision
from matplotlib import pyplot as plt

from Utils.shortcuts import get_logger

plt.switch_backend("Agg")


def build_logger(folder=None, args=None, logger_name=None):
    FORMAT = "%(asctime)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT)
    logger = get_logger(logger_name)
    # logger.setLevel(logging.DEBUG)

    if folder is not None:
        fh = logging.FileHandler(
            filename=os.path.join(
                folder, "logfile{}.log".format(time.strftime("%m-%d"))
            )
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s;%(levelname)s|%(message)s", "%H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(
        level=logging.INFO, fmt=FORMAT, datefmt=DATEF, level_styles=LEVEL_STYLES
    )

    def get_list_name(obj):
        if type(obj) is list:
            for i in range(len(obj)):
                if callable(obj[i]):
                    obj[i] = obj[i].__name__
        elif callable(obj):
            obj = obj.__name__
        return obj

    sorted_list = sorted(args.items(), key=operator.itemgetter(0))
    host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
    logger.info("#" * 120)
    logger.info("----------Configurable Parameters In this Model----------")
    logger.info(host_info)
    for name, val in sorted_list:
        logger.info("# " + ("%30s" % name) + ":\t" + str(get_list_name(val)))
    logger.info("#" * 120)
    return logger


class Logger(object):
    def __init__(self, log_dir="./logs"):
        self.stats = dict()
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}
        if k not in self.stats[category]:
            self.stats[category][k] = []
        self.stats[category][k].append((it, v))
        # self.print_fn("Itera {}, {}'s {} is {}".format(it, category, k, v))

    def add_imgs(self, imgs, name=None, class_name=None, vrange=None):
        if class_name is None:
            outdir = self.log_dir
        else:
            outdir = os.path.join(self.log_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if isinstance(name, str):
            outfile = os.path.join(outdir, "{}.png".format(name))
        else:
            outfile = os.path.join(outdir, "%08d.png" % name)

        if vrange is None:
            maxv, minv = float(torch.max(imgs)), float(torch.min(imgs))
        else:
            maxv, minv = max(vrange), min(vrange)
        imgs = (imgs - minv) / (maxv - minv + 1e-8)
        # print(torch.max(imgs), torch.min(imgs))
        imgs = torchvision.utils.make_grid(imgs)
        torchvision.utils.save_image(imgs, outfile, nrow=8)

    def get_last(self, category, k, default=0.0):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename=None):
        if filename is None:
            filename = "stat.pkl"
        filename = os.path.join(self.log_dir, filename)
        with open(filename, "wb") as f:
            pickle.dump(self.stats, f)
