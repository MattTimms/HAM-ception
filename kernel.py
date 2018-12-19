#!/usr/bin/env python
# coding: utf-8

# # **Dermatology Pigmented Lesion Classification**
#
# By: Alex Liu
#
# The HAM10000 dataset (https://arxiv.org/abs/1803.10417) was published in spring of this year.  In the words of the authors:
#
# > We collected dermatoscopic images from different populations acquired and stored by different modalities. Given this diversity we had to apply different acquisition and cleaning methods and developed semi-automatic workflows utilizing specifically trained neural networks. The final dataset consists of 11788 dermatoscopic images, of which 10010 will be released as a training set for academic machine learning purposes and will be publicly available through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions. More than 50% of lesions have been confirmed by pathology, the ground truth for the rest of the cases was either follow-up, expert consensus, or confirmation by in-vivo confocal microscopy.
#
#  I created this notebook to practice both exploratory data analysis and also to create my first deep convolutional network using Kera.
#
#  Gratitude to:
#  Kevin Mader (Kaggle) for uploading this dataset and providing instructions for loading the images.

# ## **Initialization**

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# Todo: At some point it might be useful to use another framework like Pytorch and see how that compares to Tensorflow.


def main():
    base_skin_dir = os.path.join('dataset/skin-cancer-mnist-ham10000')

    # Merge images from both folders into one dictionary

    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    # This dictionary is useful for displaying more human-friendly labels later on

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }


    # Read in the csv of metadata

    tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

    # Create some new columns (path to image, human-readable name) and review them

    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
    tile_df.sample(5)


    # ## **Exporatory Data Analysis**


    # Get general statistics for the dataset

    tile_df.describe(exclude=[np.number])



    # Let's see the distribution of different cell types

    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)


    # Too many melanocytic nevi - let's balance it a bit!

    tile_df = tile_df.drop(tile_df[tile_df.cell_type_idx == 4].iloc[:5000].index)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)


    # **Import and resize images**


    input_dims = (50, 50)
    input_shape = input_dims + (3,)


    # Load in all of the images into memory - this will take a while.
    # We also do a resize step because the original dimensions of 450 * 600 * 3 was too much for TensorFlow

    tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize(input_dims)))


    # **Seeing is believing...**


    n_samples = 5
    fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
    for n_axs, (type_name, type_rows) in zip(m_axs,
                                             tile_df.sort_values(['cell_type']).groupby('cell_type')):
        n_axs[0].set_title(type_name)
        for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
            c_ax.imshow(c_row['image'])
            c_ax.axis('off')
    fig.savefig('category_samples.png', dpi=300)



    # See the image size distribution - should just return one row (all images are uniform)
    tile_df['image'].map(lambda x: x.shape).value_counts()


    # ## **Deep Convolutional Classifier**
    #
    # We are going to build a multi-class classifier using a deep convolutional network architecture.

    # **Let's create our training and test sets.**


    # Shuffle the initial dataset
    tile_df = tile_df.sample(frac=1)

    # Training and test set division
    train = tile_df[:-500]
    test = tile_df[-500:]


    # **Assign the data structures we are going to pass into the classifier.**


    x_train = np.asarray(train['image'].tolist()).reshape((train.shape[0],)+input_shape) #/ 255
    x_test = np.asarray(test['image'].tolist()).reshape((test.shape[0],)+input_shape) #/ 255

    '''
    x_train_mean = np.mean(x_train, axis=0)
    x_train_mean_reptrain = np.repeat(x_train_mean[np.newaxis,:,:,:], x_train.shape[0], axis=0)
    x_train_mean_reptest = np.repeat(x_train_mean[np.newaxis,:,:,:], x_test.shape[0], axis=0)
    
    x_train = (x_train - x_train_mean_reptrain)
    x_test = (x_test - x_train_mean_reptest)
    '''

    y_train = train['cell_type_idx']
    y_test = test['cell_type_idx']

    # Perform one-hot encoding on the labels
    y_train = to_categorical(y_train, num_classes = 7)
    y_test = to_categorical(y_test, num_classes = 7)


    # ** Create image augmentor **

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True)
    datagen.fit(x_train)



    n_dim = 5
    n_samples = n_dim**2
    flower = datagen.flow(x_train[51:52], y_train[51:52], batch_size=1)

    fig, all_axs = plt.subplots(n_dim, n_dim, figsize = (4*n_dim, 3*n_dim))
    for row_axs in all_axs:
        for one_ax in row_axs:
            one_image = flower.next()[0][0]
            one_image[one_image>1.0] = 1.0
            one_image[one_image<0.0] = 0.0
            one_ax.imshow(one_image)
            one_ax.axis('off')


    # **Define our deep convolutional network architecture**

    num_classes = 7

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



    batch_size = 32
    real_data_epochs = 12

    x_train_std = datagen.standardize(x_train.astype('float'))
    x_test_std = datagen.standardize(x_test.astype('float'))

    history = model.fit(x_train_std, y_train,
            batch_size=batch_size,
            epochs=real_data_epochs,
            verbose=1,
            validation_data=(x_test_std, y_test))



    gen_data_epochs = 24

    # For some reason we get an error unless we start with model.fit before model.fit_generator
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) // gen_data_epochs,
            epochs=gen_data_epochs,
            verbose=1,
            validation_data=(x_test_std, y_test))

    score = model.evaluate(x_test_std, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # ## Model Validation

    # **Loss and accuracy curves for training and test sets.**

    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)


    # **Confusion matrix**

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Predict the values from the validation dataset
    y_pred = model.predict(x_test_std)
    # Convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred,axis = 1)
    # Convert validation observations to one hot vectors
    y_true = np.argmax(y_test,axis = 1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(7))


    # Now let's see how much of each class is incorrect.


    label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    plt.bar(np.arange(7),label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')

if __name__ == '__main__':
    main()