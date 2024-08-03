'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''

import random
import os.path as osp

import time
import numpy as np
import torch
import logging

from utils.default_config import _C as default_cfg

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_cfg(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def setup_cfg(config_file = None):
    cfg =  default_cfg.clone()
    if config_file is not None:
        cfg.merge_from_file(config_file)
    return cfg

def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    logging.basicConfig(filename=fpath, level=logging.DEBUG) 
