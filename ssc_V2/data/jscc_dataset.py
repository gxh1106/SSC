from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data
NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95

from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.data.transforms import augment

import random

from basicsr.utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class HR_image(data.Dataset):
    """Read only gt images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(HR_image, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [os.path.join(self.gt_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        gt_size = self.opt.get('gt_size', 256)
        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            img_gt = self._random_crop(img_gt, gt_size)
            # flip, rotation
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            # img_gt = img_gt[0:gt_size, 0:gt_size, :] # 这是左上角裁剪
    
            # 改为中心裁剪
            h, w, _ = img_gt.shape
            top = (h - gt_size) // 2
            left = (w - gt_size) // 2
            img_gt = img_gt[top:top + gt_size, left:left + gt_size, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
    
    def _random_crop(self, img_gts, gt_patch_size):

        if not isinstance(img_gts, list):
            img_gts = [img_gts]

        # determine input type: Numpy array or Tensor
        input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

        if input_type == 'Tensor':
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            h_gt, w_gt = img_gts[0].shape[0:2]

        # randomly choose top and left coordinates for lq patch
        top_gt = random.randint(0, h_gt - gt_patch_size)
        left_gt = random.randint(0, w_gt - gt_patch_size)

        # crop corresponding gt patch
        if input_type == 'Tensor':
            img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
        else:
            img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
        if len(img_gts) == 1:
            img_gts = img_gts[0]

        return img_gts
    

@DATASET_REGISTRY.register()
class KodakDataset(data.Dataset):
    """Dataset for testing on Kodak24.

    This version implements the user's specified cropping logic:
    It performs a center crop on the image, rounding the dimensions down
    to the nearest multiple of a specified divisor (e.g., 128).

    Args:
        opt (dict): Config for the dataset. It contains the following keys:
            dataroot_gt (str): Data root path for ground-truth images.
            crop_divisor (int): The number to which dimensions are rounded down.
                                Default is 128.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(KodakDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.gt_folder = opt['dataroot_gt']
        
        # 从配置中获取裁剪的整除数，默认为 128
        self.crop_divisor = opt.get('crop_divisor', 128)

        image_extensions = ('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff')
        self.paths = sorted(list(scandir(self.gt_folder, suffix=image_extensions, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt_numpy = imfrombytes(img_bytes, float32=True)

        # 1. 获取原始图像尺寸
        original_h, original_w, _ = img_gt_numpy.shape

        # 2. 计算向下取整的目标尺寸
        target_h = original_h - original_h % self.crop_divisor
        target_w = original_w - original_w % self.crop_divisor

        # 3. 执行中心裁剪 (在 numpy 数组上)
        top = (original_h - target_h) // 2
        left = (original_w - target_w) // 2
        cropped_img = img_gt_numpy[top:top + target_h, left:left + target_w, ...]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(cropped_img, bgr2rgb=True, float32=True)

        # Normalize if needed
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'gt': img_gt,
            'gt_path': gt_path
            # 【注意】这里不再需要返回 'original_size'，因为图像已经被裁剪
        }

    def __len__(self):
        return len(self.paths)
    

@DATASET_REGISTRY.register()
class HR_image0(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, dataset_opt):
        self.opt = dataset_opt
        self.imgs = []
        for dir in dataset_opt['dataroot_gt']:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = dataset_opt.get('img_dims', [3, 256, 256])
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            # transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)

@DATASET_REGISTRY.register()
class Datasets(Dataset):
    def __init__(self, dataset_opt):
        self.data_dir = dataset_opt['dataroot_gt']
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img, name
    def __len__(self):
        return len(self.imgs)

@DATASET_REGISTRY.register()
class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)

    def __len__(self):
        return self.len * 10


def get_loader(args, config):
    if args.trainset == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)
        # test_dataset = HR_image(config, config.test_data_dir)
    elif args.trainset == 'CIFAR10':
        dataset_ = datasets.CIFAR10
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        train_dataset = dataset_(root=config.train_data_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=False)

        test_dataset = dataset_(root=config.test_data_dir,
                                train=False,
                                transform=transform_test,
                                download=False)

        train_dataset = CIFAR10(train_dataset)

    else:
        train_dataset = Datasets(config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True,
                                               drop_last=True)
    if args.trainset == 'CIFAR10':
        test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1024,
                                  shuffle=False)

    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader

