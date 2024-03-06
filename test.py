import torch
import time
import json

from pathlib import Path

from model.model import Model
from model.losses import Losses
from dataset import ICData, collate_fn
from evaluate import evaluate

device = torch.device('cuda:0')

n_epoch = 1
batch_size = 4
use_cache = True
learning_rate = 1e-6

model = Model()
model.load_state_dict(torch.load('icmn_checkpoints_best.pth')) 
model.setDevice(device)

types = json.loads(Path('type_dict.json').read_text())

test_data = ICData(model.spotting, 
                    data_dir= 'data/test_img', 
                    label_dir= 'data/test_label', 
                    target_size = 256, 
                    device = device,
                    direct_aug = False,
                    use_cache = use_cache)
test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                pin_memory=True,
                collate_fn=collate_fn)


criterion = Losses(0.5, n_epoch)
criterion.setDevice(device)
evaluation = evaluate(model, test_loader, criterion, batch_size, types = types)

print(evaluation)