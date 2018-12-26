from __future__ import print_function, division
import os
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from common.data_loader import import_ham_dataset
from common.tensorboard_logger import TensorboardLogger

from torch.optim import lr_scheduler
import torchvision.models as models
import time

from torch.autograd import Variable
from tqdm import tqdm


# Hyper parameters
batch_size = 32
num_epochs = 25

logger_tensorboard = TensorboardLogger(log_dir='./')

def main():
    #
    model = models.inception_v3(pretrained=True)

    dir_dataset = './dataset/skin-cancer-mnist-ham10000/'
    dataset = import_ham_dataset(dataset_root=dir_dataset, training=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    n_class = 6  # pull from dataset output classes

    # Print layers
    for name, child in model.named_children():
        for name2, params in child.named_parameters():
            print(name, name2, 'frozen=%r' % params.requires_grad)

    # Freeze all layers
    for params in model.parameters():
        params.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)

    # # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
    # ct = []
    # for name, child in model.named_children():
    #     if "Conv2d_4a_3x3" in ct:
    #         for params in child.parameters():
    #             params.requires_grad = True
    #     ct.append(name)


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    ## Training
    from training import train_model
    model = train_model(model, dataloader, len(dataset), criterion, optimizer, scheduler, device, logger_tensorboard, num_epochs=10)


    model_save_loc = "test" + ".pth"
    torch.save(model.state_dict(), model_save_loc)


if __name__ == '__main__':
    args = {
        'training': True,
        'path_model': '1.pth',
        'gpu': True,
    }
    main()
