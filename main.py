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
from training import train_model, test_model

from torch.optim import lr_scheduler
import torchvision.models as models
import time

from torch.autograd import Variable
from tqdm import tqdm

# Hyper parameters
BATCH_SIZE = 32
NUM_EPOCHS = 25

logger_tensorboard = TensorboardLogger(log_dir='./')


def main(path_dataset, training, path_model, use_gpu):
    """

    Args:
        path_dataset (str): directory path to dataset.
        training (bool): training or testing mode.
        path_model (str): file path to load/save model.
        use_gpu (bool): use GPU if available.
    """
    # Import dataset
    dataset = import_ham_dataset(dataset_root=path_dataset, training=training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    n_class = dataset.NUM_CLASS

    # Load InceptionV3 network
    model = models.inception_v3(pretrained=True)

    # Print network layer architecture
    for name, child in model.named_children():
        for name2, params in child.named_parameters():
            print(name, name2, 'frozen=%r' % params.requires_grad)

    # Freeze all layers
    for params in model.parameters():
        params.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)  # Move model to device

    # Model training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # # Training
    if training:
        model = train_model(model, dataloader, len(dataset), criterion, optimizer, scheduler, device,
                            logger_tensorboard, num_epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), path_model)

    # # Testing
    model.load_state_dict(torch.load(path_model))
    test_model(model, dataloader, len(dataset), criterion, device)


if __name__ == '__main__':
    args = {
        'path_dataset': './dataset/skin-cancer-mnist-ham10000/',
        'training': True,
        'path_model': 'ham_model.pth',
        'use_gpu': True,
    }
    main(**args)
