import copy
import logging
import os
import random
import time

from git import Repo
import numpy as np
import torch
import yaml
from easydict import EasyDict


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self
def get_version(path='./'):
    repo = Repo(path)
    return repo.active_branch.name, repo.head.object.hexsha

def load_config(path):
    with open(path, 'r') as f:
        return to_easydict(yaml.safe_load(f))


def to_easydict(obj):
    if isinstance(obj, EasyDict):
        return EasyDict({k: to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, dict):
        return EasyDict({k: to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_easydict(v) for v in obj]
    return obj


def merge_easydict(base, override):
    merged = to_easydict(copy.deepcopy(base))
    override = to_easydict(override)
    for key, value in override.items():
        if isinstance(value, EasyDict) and isinstance(merged.get(key), EasyDict):
            merged[key] = merge_easydict(merged[key], value)
        else:
            merged[key] = to_easydict(copy.deepcopy(value))
    return merged


def load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_dataset_config(runtime_config, checkpoint_config, data_path=None, split_path=None):
    dataset_config = EasyDict()
    if 'data' in checkpoint_config and checkpoint_config.data is not None:
        dataset_config = to_easydict(copy.deepcopy(checkpoint_config.data))
    if 'data' in runtime_config and runtime_config.data is not None:
        dataset_config = merge_easydict(dataset_config, runtime_config.data)
    if data_path:
        dataset_config.path = data_path
    if split_path:
        dataset_config.split = split_path
    return dataset_config

def has_changes(path='./'):
    repo = Repo(path)
    changed_files = [f.a_path for f in repo.index.diff(None)] + repo.untracked_files
    changed_files = list(filter(lambda p: not p.startswith('configs/'), changed_files))
    if len(changed_files) > 0:
        print('\n\nYou have uncommitted changes:')
        for fn in changed_files:
            print(' - %s' % fn)
        print('Please commit your changes before running the script.\n\n')
        return True
    else:
        return False

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    
    log_dir = os.path.join(root, fn)
    count = 0
    while os.path.exists(log_dir):
        log_dir = os.path.join(log_dir, f"{prefix}_{fn}_{count}")
        count += 1
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
