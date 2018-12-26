"""InceptionV3 w/ a side of HAM
Written by Matthew Timms for DeepNeuron-AI

Image classification of HAM10000 dataset using pre-trained InceptionV3.

Usage:
      main.py [options]
      main.py (-h | --help)
"""
from __future__ import print_function, division
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torchvision.models as models

from common.data_loader import import_ham_dataset
from common.tensorboard_logger import TensorboardLogger
from training import train_model, test_model


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./dataset/skin-cancer-mnist-ham10000/', help='path to dataset')
parser.add_argument('--training', action='store_true', help='train model')
parser.add_argument('--model_path', default='./ham_model.pth', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables CUDA and GPU usage')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--epochs', type=int, help='number of training epochs', default=10)
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
opt = parser.parse_args()


def main():
    # Import dataset
    dataset = import_ham_dataset(dataset_root=opt.dataroot, training=opt.training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                             num_workers=opt.workers)
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
    logger_tensorboard = TensorboardLogger(log_dir=opt.model_path)

    # # Training
    if opt.training:
        model = train_model(model, dataloader, len(dataset), criterion, optimizer, scheduler, device,
                            logger_tensorboard, num_epochs=opt.epochs)
        torch.save(model.state_dict(), opt.model_path)

    # # Testing
    else:
        model.load_state_dict(torch.load(opt.model_path))
        test_model(model, dataloader, len(dataset), criterion, device)


if __name__ == '__main__':
    main()
