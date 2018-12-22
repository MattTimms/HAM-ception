import os
from skimage import io, transform
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HAM10000(Dataset):
    """HAM10000 dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.ham_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ham_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.ham_frame.iloc[idx, 1], '.jpg')
        image = io.imread(img_name)
        meta = self.ham_frame.iloc[0, :].drop('image_id').as_matrix()
        sample = {'image': image, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample


def import_ham10000(dataset_root: str):
    """
    Returns dataset class instance for DataLoader. Downloads dataset if not present in dataset_root.
    """
    dataset = HAM10000(
        csv_file='HAM10000_metadata.csv',
        root_dir=dataset_root,
        transform=transforms.Compose([
            transforms.Resize(244),  # will maintain aspect ratio if single int given
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # todo triple check vals from model source paper
        ])
    )
    return dataset



