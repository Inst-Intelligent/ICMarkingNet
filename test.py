'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''


import torch
import time
import argparse

from model import Model
from losses import Losses
from dataset import ICData, collate_fn
from evaluate import evaluate, print_results
from utils.config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default="", help="path to configuration")
    args = parser.parse_args()
    cfg = setup_cfg(args.config if args.config else None)
    print_cfg(cfg)

    # fix random seed
    if cfg.SEED > -1:
        set_seed(cfg.SEED)

    # set cpu or cuda device
    if torch.cuda.is_available() and cfg.DEVICE !="cpu":
        device = torch.device(cfg.DEVICE)
    else:
        device = torch.device("cpu")

    print("Building model")
    model = Model()
    model.load_state_dict(torch.load(cfg.TEST.CHECKPOINT))
    model.setDevice(device)
    print(f"The model has been loaded on device {cfg.DEVICE}.")

    print("Building dataset")
    test_data = ICData(model.spotting, 
                    data_dir = cfg.TEST.IMG_PATH, 
                    label_dir= cfg.TEST.LABEL_PATH, 
                    device = device,
                    direct_aug = cfg.TEST.AUGMENT_DATA)
    test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=cfg.TEST.BATCH_SIZE,
                    shuffle=cfg.TEST.SHUFFLE_DATA,
                    num_workers=cfg.TEST.NUM_WORKERS,
                    drop_last=False,
                    pin_memory=True,
                    collate_fn=collate_fn)
    print("Dataset has been setup")

    criterion = Losses(sigma = cfg.LOSS.SIGMA,
                       alpha = cfg.LOSS.ALPHA)
    criterion.setDevice(device)

    total_time = 0
    st = time.time()
    print("Start evaluation")
    evaluation = evaluate(model, test_loader, criterion)
    total_time += time.time() - st
    print_results(evaluation)
