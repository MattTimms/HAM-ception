"""InceptionV3 w/ a side of HAM
Written by Matthew Timms for DeepNeuron-AI

Image classification of HAM10000 dataset using pre-trained InceptionV3.

Usage:
      main.py [options]
      main.py (-h | --help)
"""
from __future__ import print_function, division
import argparse
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torchvision.models as models
from tensorboardX import SummaryWriter

from common.data_loader import import_ham_dataset
from training import train_model, test_model


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./../data/skin-cancer-mnist-ham10000/', help='path to dataset')
parser.add_argument('--testing', default=False, help='path to trained model for evaluation [default: False]')
parser.add_argument('--outf', default='./saved/runs/', help='folder to working/output directory [default: saved/runs/]')
parser.add_argument('--cuda', action='store_true', help='enables CUDA and GPU usage')
parser.add_argument('--workers', type=int, help='number of data loading workers [default: 0]', default=0)
parser.add_argument('--epochs', type=int, help='number of training epochs [default: 10]', default=10)
parser.add_argument('--batch_size', type=int, default=32, help='input batch size [default: 32]')


def main(opt):
    # Check you can write to output path directory
    testing = opt.testing if not opt.testing else True

    if not testing:
        opt.outf = os.path.join(opt.outf, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        os.makedirs(opt.outf)
        if not os.access(opt.outf, os.W_OK):
            raise OSError("--outf is not a writeable path: %s" % opt.outf)
    else:
        opt.outf = os.path.split(opt.testing)[0]

    # Import dataset
    dataset = import_ham_dataset(dataset_root=opt.dataroot, training=not testing, outf=opt.outf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                             num_workers=opt.workers)
    n_class = dataset.NUM_CLASS

    # Load InceptionV3 network
    model = models.inception_v3(pretrained=True)

    # Freeze all layers
    for params in model.parameters():
        params.requires_grad = False

    # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
    ct = []
    for name, child in model.named_children():
        if "Conv2d_4a_3x3" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)

    # Print network layer architecture
    for name, child in model.named_children():
        for name2, params in child.named_parameters():
            print(name, name2, 'trainable=%r' % params.requires_grad)

    if opt.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Using", device)
    model.to(device)  # Move model to device

    # Model training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initiate TensorBoard logger
    logger_tensorboard = SummaryWriter(opt.outf)

    # # Training
    if not testing:
        train_model(model, dataloader, len(dataset), criterion, optimizer, scheduler, device, opt.outf,
                    logger_tensorboard, num_epochs=opt.epochs)

    # # Testing
    else:
        model.load_state_dict(torch.load(os.path.join(opt.testing), map_location=device))
        test_model(model, dataloader, len(dataset), criterion, device)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)
