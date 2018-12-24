import os
from PIL import Image
import pandas as pd
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


class HAM10000(Dataset):
    """HAM10000 dataset."""
    def __init__(self, csv_file, root_dir, transform=None, minimal=True):
        self.ham_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.minimal = minimal
        self.dict = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df':  6,
        }

    def __len__(self):
        return len(self.ham_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'HAM10000_images_part_1', self.ham_frame.iloc[idx, 1] + '.jpg')
        if not os.path.exists(img_name):
            img_name = os.path.join(self.root_dir, 'HAM10000_images_part_2', self.ham_frame.iloc[idx, 1] + '.jpg')

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        meta = self.ham_frame.iloc[0, :].drop('image_id').values.tolist()
        if self.minimal:
            meta = self.dict[meta[1]]

        return image, meta


def import_ham10000(dataset_root: str):
    """
    Returns dataset class instance for DataLoader. Downloads dataset if not present in dataset_root.
    """
    dataset = HAM10000(
        csv_file='HAM10000_metadata.csv',
        root_dir=dataset_root,
        transform=transforms.Compose([
            transforms.Resize(299),  # will maintain aspect ratio if single int given
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    return dataset



