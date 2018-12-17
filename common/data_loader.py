import torchvision.datasets as dset
import torchvision.transforms as transforms
import shutil
import os


class DatasetLoadError(Exception):
    pass


def import_dataset(dataset_name: str, dataset_root: str, input_image_size: int, training=True):
    """
    Returns dataset class instance for DataLoader. Downloads dataset if not present in dataset_root.
    """
    if dataset_name in ['folder', 'imagefolder']:
        # folder dataset
        dataset = dset.ImageFolder(
            root=dataset_root,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),  # will maintain aspect ratio if single int given
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # by colour channels
            ])
        )
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(
            root=dataset_root,
            download=True,
            train=training,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif dataset_name == 'lsun':
        print("[!] LSUN dataset downloads are between 2.3 - 43.0 GB.")
        dataset = dset.LSUN(
            root=dataset_root,
            classes=['bedroom_train'],
            transform=transforms.Compose([
                transforms.Resize(input_image_size),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(
            root=dataset_root,
            download=True,
            transform=transforms.Compose([
               transforms.Resize(input_image_size),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif dataset_name == 'birds':
        dataset_folder = os.path.join(dataset_root, 'CUB_200_2011/images/')
        if not os.path.exists(dataset_folder):
            raise DatasetLoadError(
                "Dataset has not been downloaded.\n"
                "Download from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html")

        # folder dataset
        dataset = dset.ImageFolder(
            root=dataset_folder,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),  # will maintain aspect ratio if single int given
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # by colour channels
            ])
        )
    elif dataset_name == 'flowers':
        dataset_folder = os.path.join(dataset_root, 'flowers/')
        if not os.path.exists(dataset_folder):
            raise DatasetLoadError(
                "Dataset has not been downloaded.\n"
                "Download from http://www.robots.ox.ac.uk/~vgg/data/flowers/17/")

        # folder dataset
        dataset = dset.ImageFolder(
            root=dataset_folder,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),  # will maintain aspect ratio if single int given
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # by colour channels
            ])
        )
    elif dataset_name == "pokemon":
        dataset_folder = os.path.join(dataset_root, 'sprites-master/')
        if not os.path.exists(dataset_folder):
            print("Downloading pokemon dataset...")
            os.system(
                "git clone https://github.com/PokeAPI/sprites.git %s" % os.path.join(dataset_root, 'sprites-master/'))
            shutil.rmtree(os.path.join(dataset_root, "sprites-master/sprites/items/"))
            img_dir = os.path.join(dataset_root, "sprites-master/sprites/pokemon/")
            all_subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
            for dir in all_subdirs:
                print(os.path.join(img_dir, dir))
                shutil.rmtree(os.path.join(img_dir, dir))
            print("Dataset downloaded.")

        # folder dataset
        dataset = dset.ImageFolder(
            root=dataset_folder,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),  # will maintain aspect ratio if single int given
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # by colour channels
            ])
        )
    else:
        raise DatasetLoadError("dataset_name %s is not supported." % dataset_name)
    return dataset
