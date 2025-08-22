import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_loader
from utils import *
import numpy as np # MODIFIED: Added missing import for numpy
torch.backends.cudnn.benchmark = True
# MODIFIED: Commented out to allow visibility of all GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision
import os # MODIFIED: Added missing import for os

parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='ffhq',
                    choices=['kodak', 'CLIC21', 'ffhq'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                    help='SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=str, default='96',
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='10',
                    help='random or fixed snr')
parser.add_argument('--model_size', type=str, default='base',
                    choices=['small', 'base', 'large'], help='SwinJSCC model size')
args = parser.parse_args()

class config():
    seed = 42
    pass_channel = True
    CUDA = True
    # MODIFIED: Using cuda if available, regardless of the specific device number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7].replace(":", "-") # MODIFIED: Replaced ':' for Windows compatibility
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        # MODIFIED: Updated dataset path
        train_data_dir = "./datasets/CIFAR10/"
        test_data_dir = "./datasets/CIFAR10/"
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
        save_model_freq = 100
        image_dims = (3, 256, 256)
        # MODIFIED: Updated base path for datasets
        base_path = "./datasets/"
        if args.testset == 'kodak':
            # MODIFIED: Updated test data path
            test_data_dir = [base_path + "kodak/"]
        elif args.testset == 'CLIC21':
            # MODIFIED: Updated test data path
            test_data_dir = [base_path + "clic2021/test/"]
        elif args.testset == 'ffhq':
            # MODIFIED: Updated test data path
            test_data_dir = [base_path + "ffhq/"]
        
        # MODIFIED: Updated training data paths to reflect the new base_path
        train_data_dir = [base_path + 'clic2020/**',
                          base_path + 'clic2021/train',
                          base_path + 'clic2021/valid',
                          base_path + 'clic2022/val',
                          base_path + 'DIV2K_train_HR',
                          base_path + 'DIV2K_valid_HR']
        batch_size = 16
        downsample = 4
        if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
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
        elif args.model_size =='large':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )

if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    # MODIFIED: When using DataParallel, model state_dict keys are prefixed with 'module.'
    # We need to handle this when loading weights, especially for testing on a single GPU.
    # A simple way is to create a new state_dict without the prefix.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrained.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=True)
    del pretrained

def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to(config.device) # MODIFIED: Use device from config
            recon_image, CBR, SNR, mse, loss_G = net(input)
            
            # MODIFIED: Handle loss from DataParallel which returns a tensor for each GPU
            if isinstance(loss_G, torch.Tensor):
                loss_G = loss_G.mean()
            if isinstance(CBR, torch.Tensor):
                CBR = CBR.mean().item()
            if isinstance(SNR, torch.Tensor):
                SNR = SNR.mean().item()
            if isinstance(mse, torch.Tensor):
                mse = mse.mean()

            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log10(1. / mse)) # MODIFIED: Original code assumed 255 range, but images are typically normalized to [0,1]
                psnrs.update(psnr.item())
                msssim = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item() # MODIFIED: 1 - mssim is loss, mssim is the metric
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(1)

            if (global_step % config.print_step) == 0:
                process = (batch_idx + 1) / len(train_loader) * 100.0 # MODIFIED: Corrected progress calculation
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{(batch_idx + 1)}/{len(train_loader)}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.4f} ({msssims.avg:.4f})', # MODIFIED: Changed format
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                # MODIFIED: Clear metrics after logging, not before loop
                # for i in metrics:
                #     i.clear()
    else: # For DIV2K
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to(config.device) # MODIFIED: Use device from config
            recon_image, CBR, SNR, mse, loss_G = net(input)

            # MODIFIED: Handle loss from DataParallel which returns a tensor for each GPU
            if isinstance(loss_G, torch.Tensor):
                loss_G = loss_G.mean()
            if isinstance(CBR, torch.Tensor):
                CBR = CBR.mean().item()
            if isinstance(SNR, torch.Tensor):
                SNR = SNR.mean().item()
            if isinstance(mse, torch.Tensor):
                mse = mse.mean()

            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log10(1. / mse)) # MODIFIED: Original code assumed 255 range, but images are typically normalized to [0,1]
                psnrs.update(psnr.item())
                msssim = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item() # MODIFIED: 1 - mssim is loss, mssim is the metric
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(1)

            if (global_step % config.print_step) == 0:
                process = (batch_idx + 1) / len(train_loader) * 100.0 # MODIFIED: Corrected progress calculation
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{(batch_idx + 1)}/{len(train_loader)}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.4f} ({msssims.avg:.4f})', # MODIFIED: Changed format
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
    
    # MODIFIED: Clear metrics at the end of the epoch
    for i in metrics:
        i.clear()


def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    
    # MODIFIED: Ensure recon directory exists
    recon_dir = "./data/recon"
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)

    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                if args.trainset == 'CIFAR10':
                    for batch_idx, (input, label) in enumerate(test_loader):
                        start_time = time.time()
                        input = input.to(config.device) # MODIFIED: Use device from config
                        
                        # MODIFIED: Access model using 'module' if wrapped in DataParallel
                        model_to_run = net.module if isinstance(net, nn.DataParallel) else net
                        recon_image, CBR, SNR_out, mse, loss_G = model_to_run(input, SNR, rate)

                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR_out)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log10(1. / mse))
                            psnrs.update(psnr.item())
                            msssim = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)
                        else:
                            psnrs.update(100)
                            msssims.update(1)

                else: # For DIV2K, etc.
                    for batch_idx, batch in enumerate(test_loader):
                        input, names = batch
                        start_time = time.time()
                        input = input.to(config.device) # MODIFIED: Use device from config

                        # MODIFIED: Access model using 'module' if wrapped in DataParallel
                        model_to_run = net.module if isinstance(net, nn.DataParallel) else net
                        recon_image, CBR, SNR_out, mse, loss_G = model_to_run(input, SNR, rate)
                        
                        torchvision.utils.save_image(recon_image,
                                                     os.path.join(recon_dir, f"{names[0]}"))
                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR_out)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log10(1. / mse))
                            psnrs.update(psnr.item())
                            msssim = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)
                            if msssim < 1: # Avoid log(0)
                                MSSSIM_db = -10 * np.log10(1 - msssim)
                                print(f"Image: {names[0]}, MS-SSIM(dB): {MSSSIM_db:.2f}")
                        else:
                            psnrs.update(100)
                            msssims.update(1)
            
            # Logging after each SNR/rate pair iteration
            log = (' | '.join([
                f'Testing with SNR {SNR} dB and Rate {rate}',
                f'Avg. CBR {cbrs.avg:.4f}',
                f'Avg. PSNR {psnrs.avg:.3f}',
                f'Avg. MS-SSIM {msssims.avg:.4f}',
            ]))
            logger.info(log)
            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            for t in metrics:
                t.clear()

    print("--- Test Results Summary ---")
    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    
    # Create working directories if they don't exist
    if not os.path.exists(config.workdir):
        os.makedirs(config.workdir)
    if not os.path.exists(config.models):
        os.makedirs(config.models)
    if not os.path.exists(config.samples):
        os.makedirs(config.samples)

    logger = logger_configuration(config, save_log=True if args.training else False) # MODIFIED: Only save log during training
    logger.info(args) # Log arguments
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    
    # MODIFICATION 1: Train from scratch
    # The following two lines for loading pretrained weights are commented out.
    # model_path = "/media/D/yangke/SwinJSCC/checkpoint/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
    # load_weights(model_path)
    logger.info("Starting training from scratch as no model weights are loaded.")

    # MODIFICATION 2: Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training!")
        net = nn.DataParallel(net)
    
    net = net.to(config.device) # Move model to the target device(s)

    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    
    # MODIFIED: Calculate starting epoch correctly, useful for resuming training
    steps_epoch = 0 # Start from epoch 0
    if args.training:
        logger.info("======== Starting Training ========")
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:
                logger.info(f"======== Saving model at epoch {epoch + 1} ========")
                # MODIFIED: Save state_dict from net.module if using DataParallel
                model_state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
                save_model(model_state, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                
                logger.info(f"======== Running validation at epoch {epoch + 1} ========")
                test()
                net.train() # Switch back to training mode after test
    else:
        logger.info("======== Starting Testing Only ========")
        # For testing, you must load a trained model.
        # Since the goal is to pretrain, this part requires a trained model.
        # Example of how you would load your newly trained model for testing:
        try:
            # You would replace this path with the actual path to your trained model
            # trained_model_path = "history/some_training_run/models/....model" 
            # load_weights(trained_model_path)
            # logger.info(f"Loaded weights from {trained_model_path} for testing.")
            test()
        except FileNotFoundError:
            logger.error("No model found to test. Please train a model first or provide a valid model path.")