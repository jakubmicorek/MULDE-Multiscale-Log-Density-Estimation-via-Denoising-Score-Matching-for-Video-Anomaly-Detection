import json
import os
import pickle
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile, copytree
import glob
import pathlib


def get_log_path_and_summary_writer(root_dir_runs, experiment_name, postfix=None, args=None):
    now = datetime.datetime.now()
    timestamp = "_".join(list(map(lambda x: str(x).zfill(2), [now.year, now.month, now.day, now.hour, now.minute, now.second])))
    log_path = f"{root_dir_runs}/{experiment_name}/"
    log_path += f"{timestamp}"
    if postfix is not None:
        if len(postfix) > 0:
            log_path += f"_{postfix}"
        if args.key_features.__class__ == list().__class__:
            log_path += f"_{'_'.join(args.key_features)}"
        if args.key_features.__class__ == str().__class__:
            log_path += f"_{args.key_features}"
    summary_writer = SummaryWriter(log_path)
    if args is not None:
        config_parameters = f"{'_'.join([f'{arg}_{getattr(args, arg)}' for arg in vars(args)])}"
        summary_writer.add_text('config parameters', config_parameters, 0)
    return log_path, summary_writer

def save_current_experiment_source_code(log_path):
    code_path = f"{log_path}/code/"
    # for search_folder, file_extension in [("", "*.py"), ("pre_processing/", "*.py"), ("models/", "*.py"), ("data/", "*.py")]:
    for search_folder, file_extension in [("", "*.py"), ("", "*.ipynb")]:
        file_names = glob.glob(os.path.join(search_folder, file_extension))
        if len(file_names) > 0:
            pathlib.Path(os.path.join(code_path, search_folder)).mkdir(parents=True, exist_ok=True)
            for file_name in file_names:
                copyfile(file_name, f"{code_path}{file_name}")

class LossAccumulate:
    def __init__(self):
        self.losses = dict()

    def __getitem__(self, key):
        if key not in self.losses:
            self.losses[key] = list()
        return self.losses[key]

    def __setitem__(self, key, value):
        self.losses[key] = value

    def items(self):
        return self.losses.items()

    def keys(self):
        return self.losses.keys()

    def values(self):
        return self.losses.values()