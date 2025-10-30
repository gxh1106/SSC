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

def tensor_to_numpy_bgr(tensor):
    """将 [B,C,H,W] 格式的 torch tensor 转换为 OpenCV BGR 格式的 numpy 图像"""
    numpy_img = tensor.squeeze(0).cpu().numpy()
    numpy_img = np.transpose(numpy_img, (1, 2, 0))
    numpy_img_bgr = numpy_img[:, :, [2, 1, 0]] * 255.0
    return numpy_img_bgr.astype(np.uint8)

def main():
    seed = 42
    set_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the training YAML config file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model file (.pth).')
    parser.add_argument('--input', type=str, required=True, help='Input test image file (Ground Truth), e.g., /path/to/image.png.')
    parser.add_argument('--output', type=str, default='results/error_infer_output', help='Output folder for the results.')
    parser.add_argument('--error_step', type=int, default=0, help='Error quantization step index for testing.')
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
    
    os.makedirs(args.output, exist_ok=True)

    crop_divisor = opt['datasets'].get('val_1', {}).get('crop_divisor', 128)

    # 初始化 MS-SSIM 所需的参数 (window 和 weights)
    msssim_window = create_window(11, channel=3).to(device)
    msssim_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)

    # Check if the input path is valid
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found at {args.input}")
        return
        
    img_gt_numpy = cv2.imread(args.input, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    
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
    

    with torch.no_grad():
        output = model.forward_faim(img_gt, idx_H=args.error_step)[0]
        output = torch.clamp(output, 0, 1)
        psnr_val = calculate_psnr(img_gt, output).item()
        ms_ssim_val = calculate_ms_ssim(img_gt, output, window=msssim_window, data_range=1.0, weights=msssim_weights).item()


    # 1. Create the base output filename using the original name and the custom idx
    output_base_name = os.path.splitext(os.path.basename(args.input))[0]

    # 2. Save the pre-transmission (ground truth) image
    img_gt_savename = f"{output_base_name}.png"
    img_gt_path = os.path.join(args.output, img_gt_savename)
    cv2.imwrite(img_gt_path, tensor_to_numpy_bgr(img_gt))
    print(f"  -> Saving pre-transmission image to: {img_gt_path}")

    # 3. Save the reconstructed image
    recon_savename = f"{output_base_name}_Step{args.error_step}.png"
    recon_path = os.path.join(args.output, recon_savename)
    cv2.imwrite(recon_path, tensor_to_numpy_bgr(output))
    print(f"  -> Saving reconstructed image to: {recon_path}")

    # 4. Save the metrics to a text file with the same base name
    txt_savename = f"{output_base_name}_Step{args.error_step}.txt"
    txt_path = os.path.join(args.output, txt_savename)
    with open(txt_path, 'w') as f:
        f.write(f"PSNR: {psnr_val:.4f}\n")
        f.write(f"MS-SSIM: {ms_ssim_val:.4f}\n")
    print(f"  -> Saving metrics to: {txt_path}")


if __name__ == '__main__':
    main()