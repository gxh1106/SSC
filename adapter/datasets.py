import torch
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import glob
import torchvision


class Vimeo(Dataset):
    def __init__(self, data_path:Path, if_train=True, full_mode=True):
        if not full_mode:
            assert os.path.exists(data_path / "data_list.txt"), f"data_list.txt is missing or {data_path} is error"
            data_list = "data_list.txt"
        else:
            assert os.path.exists(data_path / "data_list_full.txt"), f"data_list_full.txt is missing or {data_path} is error"
            data_list = "data_list_full.txt"
        super(Vimeo, self).__init__()
        self.if_train = if_train
        self.data_path = str(data_path.absolute())

        with open(data_path / data_list, "r") as f:
            self.input_data = f.readlines()

        print(f"ready to load {len(self.input_data)} sequences for training")

    def __len__(self):
        return len(self.input_data)
    
    def get_transform(self, train_flag):
        if train_flag:
            transform = T.Compose([
                T.RandomCrop(256, pad_if_needed=True),
                T.ToTensor()
            ])
        else:
            transform = T.Compose([T.ToTensor()])
        return transform

    def __getitem__(self, index):
        transform = self.get_transform(self.if_train)
        img_path = self.data_path + "/sequences/" + self.input_data[index].rstrip()
        img = Image.open(img_path).convert("RGB")

        img = transform(img)
        out_image = img * 2 - 1  # normalize to [-1, 1]
        
        return out_image


class Div2K(Dataset):
    """Generic DIV2K image loader.

    It recursively scans all image files under data_path when initialized.
    For training, it applies RandomCrop(crop_size) + ToTensor(); for eval, only ToTensor().
    Output is normalized to [-1, 1].
    """
    def __init__(self, data_path: Path, if_train: bool = True, crop_size: int = 256):
        super().__init__()
        self.if_train = if_train
        self.crop_size = crop_size
        self.data_path = Path(data_path)

        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for ext in exts:
            files.extend(glob.glob(str(self.data_path / "**" / ext), recursive=True))
        self.files = sorted(files)
        assert len(self.files) > 0, f"No images found under {self.data_path}"

        print(f"DIV2K: found {len(self.files)} images under {self.data_path}")

    def __len__(self):
        return len(self.files)

    def get_transform(self, train_flag: bool):
        if train_flag:
            return T.Compose([
                T.RandomCrop(self.crop_size, pad_if_needed=True),
                T.ToTensor(),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
            ])

    def __getitem__(self, index: int):
        transform = self.get_transform(self.if_train)
        img_path = self.files[index]
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img = img * 2 - 1
        return img


class CIFAR10Dataset(Dataset):
    """Wrapper for torchvision CIFAR10 to output [-1, 1] images only."""
    def __init__(self, data_path: Path, if_train: bool = True, download: bool = False):
        super().__init__()
        self.if_train = if_train
        self.base = torchvision.datasets.CIFAR10(root=str(data_path), train=if_train, transform=T.ToTensor(), download=download)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        img, _ = self.base[index]
        img = img * 2 - 1
        return img