import os
import shutil

# import socket
import pickle
import time

from Utils.checkpoints.logger import build_logger
from Utils.config import notValid
from Utils import flags

FLAGS = flags.FLAGS


def save_context(filename, config):
    FILES_TO_BE_SAVED = config["FILES_TO_BE_SAVED"]
    KEY_ARGUMENTS = config["KEY_ARGUMENTS"]

    if FLAGS.gpu.lower() not in ["-1", "none", notValid.lower()]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    default_key = ""
    for item in KEY_ARGUMENTS:
        default_key += "(" + item + "_" + str(FLAGS.__getattr__(item)) + ")"

    if FLAGS.results_folder == notValid:
        FLAGS.results_folder = "./results/"
    if FLAGS.subfolder != notValid:
        FLAGS.results_folder = os.path.join(FLAGS.results_folder, FLAGS.subfolder)
    FLAGS.results_folder = os.path.join(
        FLAGS.results_folder,
        "({file})_({data})_({time})_({default_key})_({user_key})".format(
            file=filename.replace("/", "_"),
            data=FLAGS.dataset,
            time=time.strftime("%Y-%m-%d-%H-%M-%S"),
            default_key=default_key,
            user_key=FLAGS.key,
        ),
    )

    if os.path.exists(FLAGS.results_folder):
        raise FileExistsError(
            "{} exits. Run it after a second.".format(FLAGS.results_folder)
        )

    MODELS_FOLDER = FLAGS.results_folder + "/models/"
    SUMMARIES_FOLDER = FLAGS.results_folder + "/summary/"
    SOURCE_FOLDER = FLAGS.results_folder + "/source/"

    # creating result directories
    os.makedirs(FLAGS.results_folder)
    os.makedirs(MODELS_FOLDER)
    os.makedirs(SUMMARIES_FOLDER)
    os.makedirs(SOURCE_FOLDER)
    logger = build_logger(FLAGS.results_folder, FLAGS.get_dict())
    extensions = [".py", ".yml", ".yaml"]
    for folder in FILES_TO_BE_SAVED:
        destination = SOURCE_FOLDER
        if folder != "./":
            destination += folder
            os.makedirs(destination)
        all_py_yaml_files = []
        for f in os.listdir(folder):
            for e in extensions:
                if f.endswith(e):
                    all_py_yaml_files.append(f)
        for file in all_py_yaml_files:
            shutil.copy(os.path.join(folder, file), os.path.join(destination, file))
    configs_dict = FLAGS.get_dict()
    with open(os.path.join(SOURCE_FOLDER, "configs_dict.pkl"), "wb") as f:
        pickle.dump(configs_dict, f)
    shutil.copy(FLAGS.config_file, os.path.join(SOURCE_FOLDER, "used_config.yaml.bak"))
    # Figure finished
    return logger, MODELS_FOLDER, SUMMARIES_FOLDER
