from __future__ import print_function, division
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm


def train_model(model, dataloader, dataset_size, criterion, optimizer, scheduler, device, model_path,
                logger_tensorboard, num_epochs=25):
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
            labels = Variable(metas).to(device)
            inputs = Variable(inputs).to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs

            # Calculate loss
            _, preds = torch.max(outputs.data, 1)  # get the index of the max log-probability
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            running_corrects += preds.cpu().eq(labels.cpu()).sum()  # torch.sum(preds == labels.data).data

        epoch_loss = running_loss.item() / dataset_size
        epoch_acc = running_corrects.item() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f} lr: {:.2e}\n'.format(epoch_loss, epoch_acc, optimizer.param_groups[0]['lr']))

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
            torch.save(model.state_dict(), model_path)

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
        labels = Variable(metas).to(device)
        inputs = Variable(inputs).to(device)

        # Forward pass
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs

        # Calculate loss
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.data
        running_corrects += preds.cpu().eq(labels.cpu()).sum()

    model_loss = running_loss.item() / dataset_size
    model_acc = running_corrects.item() / dataset_size

    print('Loss: {:.4f} Acc: {:.4f}\n'.format(model_loss, model_acc))
