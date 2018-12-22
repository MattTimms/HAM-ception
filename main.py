from __future__ import print_function, division
import random
import os
import numpy as np
import torch
import argparse
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from common.data_loader import import_ham10000
from common.tensorboard_logger import TensorboardLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from torch.optim import lr_scheduler
import torchvision.models as models


# load csv's
# dataload images w/ transforms
# import inception
# freeze layers
# train
# evaluate input
# evaluate output


def main():
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    model = models.inception_v3(pretrained=True)

    dir_dataset = './dataset/skin-cancer-mnist-ham10000/'
    dataset = import_ham10000(dataset_root=dir_dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(opt.workers))
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

    # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
    ct = []
    for name, child in model.named_children():
        if "Conv2d_4a_3x3" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)

    data_dir = "data/"
    input_shape = 299
    batch_size = 32
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 360
    input_shape = 244
    use_parallel = True
    use_gpu = True
    epochs = 100

    if opt.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_ft = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, use_gpu,
                           num_epochs=epochs)

    model_save_loc = args.save_loc + args.model_name + "_" + str(args.freeze_layers) + "_freeze" + "_" + str(
        args.freeze_initial_layers) + "_freeze_initial_layer" + ".pth"
    torch.save(model_ft.state_dict(), model_save_loc)




if __name__ == '__main__':
    main()
