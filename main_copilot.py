import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from net.network import SwinJSCC
from data.datasets import get_loader
from loss.distortion import MS_SSIM
from utils import AverageMeter, logger_configuration, makedirs, seed_torch


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true', help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K', choices=['CIFAR10', 'DIV2K'], help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak', choices=['kodak', 'CLIC21', 'ffhq'], help='test dataset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='training loss metric')
parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                    help='SwinJSCC variant')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], help='wireless channel model')
parser.add_argument('--C', type=str, default='96', help='bottleneck dimension list, e.g., "32,64,96"')
parser.add_argument('--multiple-snr', type=str, default='10', help='SNR list, e.g., "5,10,15"')
parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'], help='SwinJSCC size')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--save-dir', type=str, default='outputs', help='directory to save checkpoints and logs')
parser.add_argument('--resume', type=str, default='', help='path to resume checkpoint')
parser.add_argument('--num-workers', type=int, default=8, help='dataloader workers')
args = parser.parse_args()


class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    norm = False
    # logger & io
    print_step = 50
    filename = datetime.now().__str__()[:-7].replace(' ', '_').replace(':', '-')
    workdir = os.path.join(args.save_dir, f'train_{filename}')
    log = os.path.join(workdir, f'train_{filename}.log')
    samples = os.path.join(workdir, 'samples')
    models = os.path.join(workdir, 'checkpoints')
    logger = None

    # training details
    learning_rate = 1e-4
    tot_epoch = args.epochs

    # dataset defaults (paths relative to repo)
    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = os.path.join('datasets', 'CIFAR10')
        test_data_dir = os.path.join('datasets', 'CIFAR10')
        batch_size = 128
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 1
        image_dims = (3, 256, 256)
        base_path = os.path.join('datasets')
        # test directories
        if args.testset == 'kodak':
            test_data_dir = [os.path.join(base_path, 'Kodak24')]
        elif args.testset == 'CLIC21':
            test_data_dir = [os.path.join(base_path, 'DIV2K', 'DIV2K_valid_HR')]
        else:
            test_data_dir = [os.path.join(base_path, 'Kodak24')]

        # train directories (use whatever exists locally)
        train_data_dir = [
            os.path.join(base_path, 'DIV2K', 'DIV2K_train_HR'),
            os.path.join(base_path, 'DIV2K', 'DIV2K_valid_HR'),
        ]
        batch_size = 8
        downsample = 4
        if args.model in ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA']:
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        else:
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[18, 2, 2, 2], num_heads=[8, 6, 4, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )


# metric helper
if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def compute_psnr(x, y, eps=1e-8):
    # inputs in [0,1]
    mse = torch.mean((x - y).clamp(0., 1.) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


def save_checkpoint(model, optimizer, epoch, save_path):
    state = {
        'epoch': epoch,
        'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args),
    }
    torch.save(state, save_path)


def train_one_epoch(model, optimizer, train_loader, epoch, logger):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()
    msssims = AverageMeter()
    cbrs = AverageMeter()
    snrs = AverageMeter()

    end = time.time()
    for batch_idx, batch in enumerate(train_loader):
        if args.trainset == 'CIFAR10':
            inputs, _ = batch
        else:
            inputs = batch
        inputs = inputs.cuda(non_blocking=True)

        recon, cbr, snr, mse_mean, loss = model(inputs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            psnr = compute_psnr(recon.clamp(0., 1.), inputs.clamp(0., 1.))
            msssim_val = 1.0 - CalcuSSIM(inputs, recon.clamp(0., 1.)).mean()

        # update meters
        losses.update(loss.item(), inputs.size(0))
        psnrs.update(psnr.item(), inputs.size(0))
        msssims.update(msssim_val.item(), inputs.size(0))
        cbrs.update(float(cbr), 1)
        snrs.update(float(snr), 1)
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % config.print_step == 0:
            logger.info(f'Epoch [{epoch}] Step [{batch_idx+1}/{len(train_loader)}] '
                        f'Loss {losses.avg:.4f} PSNR {psnrs.avg:.2f} SSIM(1-MS) {msssims.avg:.4f} '
                        f'CBR {cbrs.avg:.4f} SNR {snrs.avg:.1f} Time {batch_time.avg:.3f}s')

    return {
        'loss': losses.avg,
        'psnr': psnrs.avg,
        'msssim_1m': msssims.avg,
        'cbr': cbrs.avg,
        'snr': snrs.avg,
    }


@torch.no_grad()
def test(model, test_loader, logger):
    model.eval()
    multiple_snr = [int(s) for s in args.multiple_snr.split(',')]
    channel_number = [int(c) for c in args.C.split(',')]

    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))

    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            psnrs = AverageMeter()
            msssims = AverageMeter()
            cbrs = AverageMeter()
            for batch in test_loader:
                if args.trainset == 'CIFAR10':
                    inputs, _ = batch
                else:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                inputs = inputs.cuda(non_blocking=True)
                recon, cbr, snr, mse_mean, loss = model(inputs, given_SNR=SNR, given_rate=rate)
                psnr = compute_psnr(recon.clamp(0., 1.), inputs.clamp(0., 1.))
                msssim_val = 1.0 - CalcuSSIM(inputs, recon.clamp(0., 1.)).mean()
                psnrs.update(psnr.item(), inputs.size(0))
                msssims.update(msssim_val.item(), inputs.size(0))
                cbrs.update(float(cbr), 1)
            results_snr[i, j] = SNR
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg

    logger.info(f"SNR: {results_snr.tolist()}")
    logger.info(f"CBR: {results_cbr.tolist()}")
    logger.info(f"PSNR: {results_psnr.tolist()}")
    logger.info(f"MS-SSIM(1-MS): {results_msssim.tolist()}")
    logger.info("Finish Test!")
    return results_psnr.mean()


if __name__ == '__main__':
    # seeding and logger
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    makedirs(config.models)
    logger.info(config.__dict__)

    # build model and data
    torch.manual_seed(seed=config.seed)
    model = SwinJSCC(args, config)

    # parallel (DataParallel)
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            logger.info(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
            model = torch.nn.DataParallel(model)
    else:
        logger.info('CUDA not available, using CPU (training will be very slow)')

    params = [{'params': model.parameters(), 'lr': config.learning_rate}]
    optimizer = optim.Adam(params, lr=config.learning_rate)

    train_loader, test_loader = get_loader(args, config)

    # optionally resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        (model.module if isinstance(model, torch.nn.DataParallel) else model).load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.training:
        best_psnr = -1.0
        for epoch in range(start_epoch, config.tot_epoch):
            stats = train_one_epoch(model, optimizer, train_loader, epoch, logger)
            # save checkpoint
            if (epoch + 1) % config.save_model_freq == 0:
                ckpt_path = os.path.join(config.models, f'epoch_{epoch+1}.pt')
                save_checkpoint(model, optimizer, epoch, ckpt_path)
                logger.info(f'Saved checkpoint to {ckpt_path}')
            # quick eval
            cur_psnr = test(model, test_loader, logger)
            best_psnr = max(best_psnr, cur_psnr)
            logger.info(f'Epoch {epoch} done. Train PSNR {stats["psnr"]:.2f}. Val PSNR {cur_psnr:.2f}. Best {best_psnr:.2f}.')
            if epoch + 1 >= args.epochs:
                break
    else:
        # test only
        test(model, test_loader, logger)

