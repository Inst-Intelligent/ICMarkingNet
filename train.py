import torch
import time
import editdistance 

from model.model import Model
from model.losses import Losses
from dataset import ICData, collate_fn
from evaluate import evaluate

device = torch.device('cuda:0')

n_epoch = 3600
batch_size = 64
use_cache = True
learning_rate = 1e-6

model = Model()
model.load_state_dict(torch.load('pretrain_icdar.pth')) 
model.setDevice(device)

train_data = ICData(model.spotting, 
                    data_dir= 'data/train_img', 
                    label_dir= 'data/train_label', 
                    target_size = 256, 
                    device = device,
                    direct_aug = True,
                    use_cache = use_cache)
train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_fn)

val_data = ICData(model.spotting, 
                    data_dir= 'data/val_img', 
                    label_dir= 'data/val_label', 
                    target_size = 256, 
                    device = device,
                    direct_aug = False,
                    use_cache = use_cache)
val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
                pin_memory=True,
                collate_fn=collate_fn)

print(len(val_data))

criterion = Losses(0.5, n_epoch)
criterion.setDevice(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=5e-4
)


total_time = 0
print('[Device]', device)

for epoch in range(n_epoch):
        st = time.time()
        batch_loss, batch_loss_angle, batch_loss_char = [], [], []
              
        model.train()
        for index, inputs in enumerate(train_loader):
                
            results, labels, words_roi = model(inputs)
            loss, details = criterion(results, labels)
            # print(loss, details)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
        print(loss, details)
        
        evaluation = evaluate(model, train_loader, criterion, batch_size)
        total_time += time.time() - st
        print(evaluation)