"""Local CPU reproduction: CNN on the MobvoiHotwords subset (2-class hotword ID).

Mirrors run.py's early-stopping loop but targets CPU (no NVIDIA GPU), the CNN
backbone, num_workers=0, and the filtered local manifests.
"""
import os
import numpy as np
import torch
from command_loader import CommandLoader
from model import CNN
from train import train, test

RES = 'data/mobvoi_hotword_dataset_resources'
BATCH = 100
EPOCHS = 5
LR = 0.005
PATIENCE = 5
CUDA = False  # no CUDA device on this machine -> train on CPU


def make_loader(name, shuffle):
    ds = CommandLoader(os.path.join(RES, name))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


if __name__ == '__main__':
    torch.manual_seed(1234)

    train_loader = make_loader('p_train_local.json', True)
    valid_loader = make_loader('p_dev_local.json', False)
    test_loader = make_loader('p_test_local.json', False)
    print('sizes  train=%d  dev=%d  test=%d' % (
        len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)), flush=True)

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs('checkpoint', exist_ok=True)
    best_valid_loss = np.inf
    iteration = 0
    epoch = 1
    while (epoch < EPOCHS + 1) and (iteration < PATIENCE):
        train(train_loader, model, optimizer, epoch, CUDA)
        valid_loss = test(valid_loader, model, CUDA)
        if valid_loss > best_valid_loss:
            iteration += 1
            print('Loss was not improved, iteration {0}'.format(iteration), flush=True)
        else:
            iteration = 0
            best_valid_loss = valid_loss
            torch.save(model, 'checkpoint/cnn_epoch{}'.format(epoch))
            print('Saving model (epoch {})...'.format(epoch), flush=True)
        epoch += 1

    print('========== FINAL TEST ==========', flush=True)
    test(test_loader, model, CUDA)
    print('========== DONE ==========', flush=True)
