import argparse
import glob

import yaml

from Utils import flags

FLAGS = flags.FLAGS
default_value = {str: "", int: -1, float: 0.0}
notValid = "NotValid_Signature"


def load_config(config_path):
    """ Loads config file.
    Args:
        config_path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(config_path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    input_arguments = []
    all_keys = FLAGS.get_dict().keys()
    # Include main configuration
    for k in all_keys:
        if k in default_arguments:
            continue
        input_arguments.append(k)
        print("Note!: input args: {} with value {}".format(k, FLAGS.__getattr__(k)))
    for k in cfg_special:
        if k in all_keys:
            print("Ignore {}".format(k))
        else:
            FLAGS.__setattr__(k, cfg_special[k])
    return input_arguments


default_arguments = {
    "config_file": "./configs/default.yml",
    "gpu": "-1",
    "key": "None",
    "dataset": notValid,
    "results_folder": notValid,
    "subfolder": notValid,
}

existed_args = list(default_arguments.keys())
flags.DEFINE_argument(
    "config_file", type=str, help="Path to config file.",
)
for k in default_arguments:
    if k != "config_file":
        flags.DEFINE_argument("-" + k, "--" + k, type=str, default=default_arguments[k])

others_yml = glob.glob("./configs/*.yml") + glob.glob("./configs/*.yaml")
for yml in others_yml:
    with open(yml, "r") as f:
        newdict = yaml.load(f, Loader=yaml.FullLoader)
    for k in newdict:
        if k in existed_args:
            continue
        existed_args.append(k)
        v = newdict[k]
        if type(v) == bool:
            flags.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
        else:
            flags.DEFINE_argument(
                "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
            )
