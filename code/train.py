import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read_data import *
from model import StackNet
from utils import Trainer

BATCH_SIZE=16
N_EPOCH = 200
SAVE_FREQ=1
TRAIN_LIST="F:/pc/changedetectiondata/Building change detection dataset256new/csv/train.csv"
VAL_LIST="F:/pc/changedetectiondata/Building change detection dataset256new/csv/val.csv"
ROOT_BE='F:/pc/changedetectiondata/Building change detection dataset256new/beforechange'
ROOT_AF='F:/pc/changedetectiondata/Building change detection dataset256new/afterchange'
ROOT_MASK='F:/pc/changedetectiondata/Building change detection dataset256new/label_change_new'
SAVE_PATH='F:/pc/job/checkpoints/job'
def get_dataloader(batch_size):
    '''mytransform = transforms.Compose([
        transforms.ToTensor()])'''

    # torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageFolder(TRAIN_LIST,ROOT_BE,ROOT_MASK
                      ),
        batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        ImageFolder(VAL_LIST,ROOT_BE,ROOT_MASK
                      ),
        batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader

def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)
    model=StackNet(n_channels=3)
    optimizer=optim.Adam(params=model.parameters())

    trainer = Trainer(model, optimizer,loss_f=nn.BCELoss ,save_freq=SAVE_FREQ,save_dir=SAVE_PATH)
    trainer.loop(N_EPOCH, train_loader, test_loader)


if __name__ == '__main__':
    main(BATCH_SIZE)