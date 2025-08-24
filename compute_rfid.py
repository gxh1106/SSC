# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import os

import torch
from torch.utils.data import Subset

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from rqvae.metrics.fid import compute_rfid, compute_psnr
from rqvae.utils.config import load_config, augment_arch_defaults


def load_model(path, ema=False):

    model_config = os.path.join(os.path.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    ckpt = torch.load(path)['state_dict_ema'] if ema else torch.load(path)['state_dict']
    model.load_state_dict(ckpt)

    return model, config


def setup_logger(result_path):
    log_fname = os.path.join(result_path, 'rfid.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == '__main__':
    """
    Computes rFID, i.e., FID between val images and reconstructed images.
    Log is saved to `rfid.log` in the same directory as the given vqvae model. 
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size to use')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--vqvae', type=str, default='', required=True,
                        help='vqvae path for recon FID')

    args = parser.parse_args()

    result_path = os.path.dirname(args.vqvae)
    logger = setup_logger(result_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    vqvae_model, config = load_model(args.vqvae)
    vqvae_model = vqvae_model.to(device)
    vqvae_model = torch.nn.DataParallel(vqvae_model).eval()
    logger.info(f'vqvae model loaded from {args.vqvae}')

    dataset_trn, dataset_val = create_dataset(config, is_eval=True, logger=logger)

    logger.info(f'#valid samples: {len(dataset_val)}')
    SUBSET_SIZE = 1000 
    attack_layer = 0
    # 确保子集大小不超过原始数据集的大小
    subset_size = min(SUBSET_SIZE, len(dataset_val))
    # 从原始验证集中创建一个固定的子集（总是取前 subset_size 个）
    dataset_val_subset = Subset(dataset_val, range(subset_size))
    logger.info(f'Using a fixed subset of {len(dataset_val_subset)} samples for quick evaluation.')
    dataset = dataset_val_subset if args.split in ['val', 'valid'] else dataset_trn

    logger.info(f'measuring PSNR on {config.dataset.type}/{args.split}')

    rfid = compute_rfid(dataset, vqvae_model, batch_size=args.batch_size, device=device)
    # avg_psnr = compute_psnr(dataset, vqvae_model, batch_size=args.batch_size, device=device, attack_layer=attack_layer)
    logger.info(f'Average PSNR : {rfid:.4f} dB')
