""" training functions
src: https://github.com/Prakashvanapalli/pytorch_classifiers/blob/master/tars/tars_training.py
"""
from __future__ import print_function, division
import time

from common.tensorboard_logger import TensorboardLogger

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def train_model(model, dataloader, dataset_size, criterion, optimizer, scheduler, device, logger_tensorboard,
                num_epochs=25):
    """

    Args:
         model (nn.Model):
         dataloader (DataLoader):
         dataset_size (int): total size of dataset
         criterion (nn._WeightedLoss):
         optimizer (Optimizer):
         scheduler (_LRScheduler):
         device (torch.device):
         logger_tensorboard (TensorboardLogger):
         num_epochs (int, optional):
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # todo dataset_size input is shit, grab from current dataloaded value or check it doesnt require the entire size

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        scheduler.step()
        model.train(True)  # Set model to training mode
        # model.train(False)

        running_loss = 0.0
        running_corrects = 0  # ttl num of correct predictions

        # Iterate over data
        for inputs, metas in tqdm(dataloader):
            labels = metas
            inputs = Variable(inputs).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            outputs = outputs.to(device)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.to(device))

            # backward + optimize in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data
            running_corrects += torch.sum(preds.cpu() == labels.data)

        epoch_loss = running_loss.item() / dataset_size
        epoch_acc = running_corrects.item() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))

        # TensorBoard Logging
        info = {
            'Loss': epoch_loss,
            'Accuracy': epoch_acc,
        }
        for tag, value in info.items():
            logger_tensorboard.scalar_summary(tag, value, epoch)

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader, dataset_size, criterion, device):
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0  # ttl num of correct predictions

    for inputs, metas in tqdm(dataloader):
        labels = metas
        inputs = Variable(inputs).to(device)

        # forward pass
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        outputs = outputs.to(device)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels.to(device))

        # statistics
        running_loss += loss.data
        running_corrects += torch.sum(preds.cpu() == labels.data)

    model_loss = running_loss / dataset_size
    model_acc = running_corrects / dataset_size

    return model_loss, model_acc
