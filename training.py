""" training functions
src: https://github.com/Prakashvanapalli/pytorch_classifiers/blob/master/tars/tars_training.py
"""
from __future__ import print_function
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm


def train_model(model, dataloader, dataset_size, criterion, optimizer, scheduler, device, num_epochs=25):
    """

    Args:
         model (nn.Model):
         dataloader (DataLoader):
         dataset_size (int):
         criterion (nn._WeightedLoss):
         optimizer (Optimizer):
         scheduler (_LRScheduler):
         device (torch.device):
         num_epochs (int):
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        scheduler.step()
        model.train(True)  # Set model to training mode
        # model.train(False)  # Set model to evaluate mode

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
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds.cpu() == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
