import os
from random import randint
import shutil

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
import torchvision.transforms as transforms


LESION_DICT = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }


class HAMDatasetException(Exception):
    pass


class HAMDataset(Dataset):
    """
    Returns dataset instance for HAM10000.

    Args:
        csv_file (str): file name of csv meta data.
        root_dir (str): directory path to dataset folder.
        training (bool): is mode training.
        transform (transforms.Compose): list of transformations to apply to input data.
        minimal (bool): return only lesion meta data.
        num_test_imgs (int): Number of test images to set aside.
    """

    NUM_CLASS = 6  # Lesion classes

    def __init__(self, csv_file, root_dir, training=True, transform=None, minimal=True, num_test_imgs=32):
        try:
            self.ham_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        except FileNotFoundError:
            raise HAMDatasetException("\nCould not read csv_file: %s\nEnsure root_dir points to dataset directory.\n"
                                      "Download HAM10000 from "
                                      "https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home" %
                                      os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.minimal = minimal

        self.num_test_imgs = num_test_imgs
        self.dir_test = os.path.join(self.root_dir, 'test')
        self.dict = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df':  6,
        }

        self.training = training
        if self.training:
            self._create_test_images()
        else:
            self._load_test_images()

    def __len__(self):
        return len(self.ham_frame)

    def __getitem__(self, idx):
        # Collect image name from csv frame
        img_name = os.path.join(self.root_dir, 'HAM10000_images_part_1', self.ham_frame.iloc[idx, 1] + '.jpg')
        if not os.path.exists(img_name):
            img_name = os.path.join(self.root_dir, 'HAM10000_images_part_2', self.ham_frame.iloc[idx, 1] + '.jpg')

        # Load image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load meta
        meta = self.ham_frame.iloc[idx, :].drop('image_id').values.tolist()
        if self.minimal:
            meta = self.dict[meta[1]]

        return image, meta

    def plot_dataset(self):
        label_counts = self.ham_frame['dx'].value_counts()
        plt.figure(figsize=(12, 6))
        sns.barplot(label_counts.index, label_counts.values, alpha=0.9)
        plt.xticks(rotation='vertical')
        plt.xlabel('Image Labels', fontsize=12)
        plt.ylabel('Counts', fontsize=12)
        plt.show()

    def _create_test_images(self):
        """
        Creates test folder populated with randomly selected samples from HAM dataset, if folder does not already exist,
        and removes them from the returned Dataset instance.
        """
        if not os.path.exists(self.dir_test):
            os.makedirs(self.dir_test)

            for i in range(self.num_test_imgs):
                # Select random image and drop from pandas ham frame
                idx = randint(0, len(self.ham_frame)-1)
                frame = self.ham_frame.iloc[idx, :]
                self.ham_frame.drop(idx, inplace=True)

                # Locate image
                img_name = frame[1] + '.jpg'
                img_path = os.path.join(self.root_dir, 'HAM10000_images_part_1', img_name)
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.root_dir, 'HAM10000_images_part_2', img_name)

                # Move image
                img_out = os.path.join(self.dir_test, frame[1] + '.jpg')
                shutil.copy(img_path, img_out)
        else:
            # Collect image names
            file_names = [os.path.splitext(fp)[0] for fp in os.listdir(os.path.join(self.dir_test))]

            # Remove image meta data from pandas frame
            for fn in file_names:
                idx = self.ham_frame.index[self.ham_frame['image_id'] == fn].tolist()
                self.ham_frame.drop(idx, inplace=True)

    def _load_test_images(self):
        if not os.path.exists(self.dir_test):
            raise HAMDatasetException("Test folder does not exist: %s" % self.dir_test)

        file_names = [os.path.splitext(fp)[0] for fp in os.listdir(os.path.join(self.dir_test))]

        if not file_names:
            raise HAMDatasetException("Test folder is empty: %s" % self.dir_test)

        # Create new pandas frame with test image meta data
        new_frame = None
        for fn in file_names:
            idx = self.ham_frame.index[self.ham_frame['image_id'] == fn].tolist()
            frame = self.ham_frame.iloc[idx, :]

            if new_frame is None:
                new_frame = frame
            else:
                new_frame = pd.concat([new_frame, frame])

        self.ham_frame = new_frame


def import_ham_dataset(dataset_root: str, training=True):
    """
    Returns dataset class instance for DataLoader. Downloads dataset if not present in dataset_root.
    """
    dataset = HAMDataset(
        csv_file='HAM10000_metadata.csv',
        root_dir=dataset_root,
        training=training,
        transform=transforms.Compose([
            transforms.Resize(299),  # required size
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # required normalisation
        ])
    )
    return dataset
