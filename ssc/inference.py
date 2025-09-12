import argparse
import cv2
import glob
import numpy as np
import os
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs import build_network
from ssc.archs.SwinSSC_arch import SwinSSC

from ssc.faim import FA_IM_Channel, FA_SISO_Channel

from addict import Dict

import random

def set_seed(seed: int):
    """
    为 PyTorch, NumPy 和 Python 的 random 模块设置随机种子以保证可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多 GPU 环境

def calculate_psnr(img1, img2, max_val=1.0):
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def create_window(window_size, channel=1):
    """创建一维高斯窗口 -> 转换为二维卷积核"""
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(X, Y, window, data_range, use_padding=False):
    """SSIM 的核心计算逻辑"""
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = F.conv2d(X, window, padding=window.shape[-1] // 2 if use_padding else 0, groups=X.shape[1])
    mu_y = F.conv2d(Y, window, padding=window.shape[-1] // 2 if use_padding else 0, groups=Y.shape[1])

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(X * X, window, padding=window.shape[-1] // 2 if use_padding else 0, groups=X.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(Y * Y, window, padding=window.shape[-1] // 2 if use_padding else 0, groups=Y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(X * Y, window, padding=window.shape[-1] // 2 if use_padding else 0, groups=X.shape[1]) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)
    
    return ssim_map.mean(dim=(1, 2, 3)), cs_map.mean(dim=(1, 2, 3))


def ssim(X, Y, window, data_range, use_padding=False):
    """多通道 SSIM 的包装函数"""
    ssim_val, cs_val = 0, 0
    for c in range(X.shape[1]):
        s, c_s = _ssim(X[:, c:c + 1], Y[:, c:c + 1], window[c:c+1], data_range, use_padding)
        ssim_val += s
        cs_val += c_s
    return ssim_val / X.shape[1], cs_val / X.shape[1]

def calculate_ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False, eps: float = 1e-8):
    weights = weights[:, None]
    levels = weights.shape[0]
    vals = []
    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)

        if i < levels - 1:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = torch.stack(vals, dim=0)
    vals = vals.clamp_min(eps)
    ms_ssim_val = torch.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], dim=0)
    return ms_ssim_val.mean()



def main():
    seed = 42
    set_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the training YAML config file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model file (.pth).')
    parser.add_argument('--input', type=str, required=True, help='Input test image folder (Ground Truth), e.g., /path/to/kodak24.')
    parser.add_argument('--output', type=str, default='results/SwinSSC_PSNR_Analysis', help='Output folder for the plot and results.')
    # parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing.')
    # parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open(args.config, 'r') as f:
            opt = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Could not load config file {args.config}. Details: {e}")
        return
        
    
    opt = Dict(opt)
    opt_args = opt.network_g.args
    image_dims = opt_args.image_dims
    channel_number = int(opt_args.C)
    if opt_args.model_size == 'small':
        opt['network_g']['encoder_kwargs'] = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        opt['network_g']['decoder_kwargs'] = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

        # opt['network_g']['rq_kwargs'] = dict(
        #     img_size=(image_dims[1], image_dims[2]),
        #     embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
        #     window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #     norm_layer=nn.LayerNorm, patch_norm=True,
        # )
    elif opt_args.model_size == 'base':
        opt['network_g']['encoder_kwargs'] = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        opt['network_g']['decoder_kwargs'] = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif opt_args.model_size =='large':
        opt['network_g']['encoder_kwargs'] = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        opt['network_g']['decoder_kwargs'] = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    
    network_config = opt['network_g'] 
    model_args = network_config.get('args')
    encoder_kwargs = network_config.get('encoder_kwargs')
    decoder_kwargs = network_config.get('decoder_kwargs')
    rq_kwargs = network_config.get('rq_kwargs')

    model = SwinSSC(
        args=argparse.Namespace(**model_args),  # 将 'args' 字典转换为 Namespace 对象
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        rq_kwargs=rq_kwargs
    )
    # 在加载权重之前，手动调用 update_resolution，强制模型采用训练时的尺寸
    #    这会重建 attn_mask，使其与 checkpoint 中的尺寸完全匹配
    downsample_ratio = 2 ** model.downsample
    H_feat, W_feat = image_dims[1] // downsample_ratio, image_dims[2] // downsample_ratio
    model.encoder.update_resolution(image_dims[1], image_dims[2])
    model.decoder.update_resolution(H_feat, W_feat)

    load_net = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    key = 'params_ema' if 'params_ema' in load_net else 'params'
    state = load_net[key]

    # 移除所有 attn_mask，避免尺寸强行匹配错误
    filtered_state = {k: v for k, v in state.items() if "attn_mask" not in k}
    removed = [k for k in state.keys() if "attn_mask" in k]

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    
    model.eval()
    model = model.to(device)

    K = 4         # 活动端口数 
    M = 64         # 星座大小 (例如，64-QAM，需要 6 比特符号)
    N = 16         # 总可用端口数
    Nr = 8        # 接收天线数
    # 定义信道物理参数
    W = 2  # 流体天线宽度为W个波长
    L_paths = 10      # 多径数
    num_H = 2     # 创建num_H个不同的信道实现

    snr_range = list(range(0, 22, 2))

    fa_im_system = FA_IM_Channel(K=K, M=M, N=N, Nr=Nr, num_H=num_H, W=W, L_paths=L_paths, device=device)

    M_fas_simo = 256  # SIMO FAS 的星座大小
    fa_simo_system = FA_SISO_Channel(M=M_fas_simo, N=N, Nr=Nr, num_H=num_H, W=W, L_paths=L_paths, device=device)

    os.makedirs(args.output, exist_ok=True)

    crop_divisor = opt['datasets'].get('val_1', {}).get('crop_divisor', 128)

    # 初始化 MS-SSIM 所需的参数 (window 和 weights)
    msssim_window = create_window(11, channel=3).to(device)
    msssim_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)

    
    psnr_results = {snr: [] for snr in snr_range}
    ms_ssim_results = {snr: [] for snr in snr_range}

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_files_in_folder = glob.glob(os.path.join(args.input, '*'))
    # 过滤出图像文件，并排序
    image_paths = sorted([p for p in all_files_in_folder if p.lower().endswith(image_extensions)])
    if not image_paths:
        print(f"Error: No images found in the input directory: {args.input}")
        return

    for path in tqdm(image_paths, desc="Processing Images"):
        
        img_gt_numpy = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        
        # 1. 获取原始尺寸
        original_h, original_w, _ = img_gt_numpy.shape

        # 2. 计算向下取整的目标尺寸
        target_h = original_h - original_h % crop_divisor
        target_w = original_w - original_w % crop_divisor

        # 3. 执行中心裁剪
        top = (original_h - target_h) // 2
        left = (original_w - target_w) // 2
        img_gt_cropped_numpy = img_gt_numpy[top:top + target_h, left:left + target_w, ...]
        
        # 4. 转换 BGR->RGB, HWC->CHW, numpy->tensor
        img_gt = torch.from_numpy(np.transpose(img_gt_cropped_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_gt = img_gt.unsqueeze(0).to(device)
        
        for snr in snr_range:
            psnr_per_channel = []
            ms_ssim_per_channel = []
            for idx_H in range(num_H):
                with torch.no_grad():
                    output = model.forward_faim(img_gt, given_SNR=snr, channel=fa_im_system, idx_H=idx_H)[0]
                    # output = model.forward_faim(img_gt, given_SNR=snr, channel=fa_simo_system, idx_H=idx_H)[0]
                    psnr_val = calculate_psnr(img_gt, output)
                    psnr_per_channel.append(psnr_val.item())
                    ms_ssim_val = calculate_ms_ssim(img_gt, output, window=msssim_window, data_range=1.0, weights=msssim_weights)
                    ms_ssim_per_channel.append(ms_ssim_val.item())

            psnr_results[snr].append(np.mean(psnr_per_channel))
            ms_ssim_results[snr].append(np.mean(ms_ssim_per_channel))

    avg_psnr_list = []
    avg_ms_ssim_list = []
    print("\n--- Average PSNR Results on Kodak24 ---")
    results_text = "SNR (dB), PSNR (dB), MS-SSIM\n"
    for snr in snr_range:
        avg_psnr = np.mean(psnr_results[snr]) if psnr_results[snr] else 0
        avg_ms_ssim = np.mean(ms_ssim_results[snr]) if ms_ssim_results[snr] else 0
        avg_psnr_list.append(avg_psnr)
        avg_ms_ssim_list.append(avg_ms_ssim)
        print(f"SNR: {snr:2d} dB  |  Avg PSNR: {avg_psnr:.4f} dB  |  Avg MS-SSIM: {avg_ms_ssim:.4f}")
        results_text += f"{snr},{avg_psnr:.4f},{avg_ms_ssim:.4f}\n"

    # 保存数值结果到 CSV 文件
    results_file_path = os.path.join(args.output, 'psnr_results.csv')
    with open(results_file_path, 'w') as f:
        f.write(results_text)
    print(f"\nNumeric results saved to {results_file_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, avg_psnr_list, marker='o', linestyle='-', label=os.path.basename(args.model_path).replace('.pth', ''))
    plt.title(f'PSNR vs. SNR Performance on Kodak24 Dataset')
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Average Peak Signal-to-Noise Ratio (PSNR) [dB]')
    plt.xticks(snr_range)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plot_path = os.path.join(args.output, 'PSNR_vs_SNR_curve.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, avg_ms_ssim_list, marker='s', linestyle='--', color='r', label=os.path.basename(args.model_path).replace('.pth', ''))
    plt.title(f'MS-SSIM vs. SNR Performance on {os.path.basename(args.input)} Dataset')
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Average MS-SSIM')
    plt.xticks(snr_range)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plot_path_msssim = os.path.join(args.output, 'MS_SSIM_vs_SNR_curve.png')
    plt.savefig(plot_path_msssim)
    print(f"MS-SSIM plot saved to {plot_path_msssim}")

def test(img_lq, model, args, window_size, snr):
    if args.tile is None:
        output = model(img_lq, given_SNR=snr)[0]
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h, w).type_as(img_lq)
        W = torch.zeros_like(E)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch, given_SNR=snr)[0]
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch)
                W[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch_mask)
        output = E.div_(W)
    return output

if __name__ == '__main__':
    main()