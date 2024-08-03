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
from evaluate import evaluate
from utils.config import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="path to configuration")
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

    n_epoch = cfg.TRAIN.NUM_EPOCH
    model = Model()

    if cfg.TRAIN.RESUME:
        model.load_state_dict(torch.load(cfg.TRAIN.RESUME))
    
    model.setDevice(device)
    
    train_data = ICData(model.spotting, 
                        data_dir = cfg.TRAIN.IMG_PATH, 
                        label_dir= cfg.TRAIN.LABEL_PATH, 
                        device = device,
                        direct_aug = cfg.TRAIN.AUGMENT_DATA)
    train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=cfg.TRAIN.SHUFFLE_DATA,
                    num_workers=cfg.TRAIN.NUM_WORKERS,
                    drop_last=False,
                    pin_memory=True,
                    collate_fn=collate_fn)


    criterion = Losses(sigma = cfg.LOSS.SIGMA,
                       alpha = cfg.LOSS.ALPHA,
                       n_iteration= n_epoch * len(train_loader))
    criterion.setDevice(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=5e-4)

    total_time = 0
    print('[Device]', device)

    iter = 0
    for epoch in range(n_epoch):
            st = time.time()
            batch_loss, batch_loss_angle, batch_loss_char = [], [], []
            
            model.train()
            for index, inputs in enumerate(train_loader):
                    
                results, labels, words_roi = model(inputs)
                loss, details = criterion(results, labels, i=iter)
                # print(loss, details)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter += 1
    
            print(loss, details)
            
            evaluation = evaluate(model, train_loader, criterion)
            total_time += time.time() - st
            logging.debug((time.time() - st))

    logging.debug(total_time)