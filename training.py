""" training functions
src: https://github.com/Prakashvanapalli/pytorch_classifiers/blob/master/tars/tars_training.py
"""
from __future__ import print_function, division
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm


def train_model(model, dataloader, dataset_size, criterion, optimizer, scheduler, device, logger_tensorboard,
                num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0  # ttl num of correct predictions

        for inputs, metas in tqdm(dataloader):
            labels = metas
            inputs = Variable(inputs).to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            outputs = outputs.to(device)

            # Calculate loss
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.to(device))

            # Backward + optimize
            loss.backward()
            optimizer.step()

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

        # Deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    # Display stats
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader, dataset_size, criterion, device):
    model.train(False)  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0  # ttl num of correct predictions

    for inputs, metas in tqdm(dataloader):
        labels = metas
        inputs = Variable(inputs).to(device)

        # Forward pass
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        outputs = outputs.to(device)

        # Calculate loss
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels.to(device))

        running_loss += loss.data
        running_corrects += torch.sum(preds.cpu() == labels.data)

    model_loss = running_loss / dataset_size
    model_acc = running_corrects / dataset_size

    print('Loss: {:.4f} Acc: {:.4f}\n'.format(model_loss, model_acc))
