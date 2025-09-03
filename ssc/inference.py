import argparse
import cv2
import glob
import numpy as np
import os
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保这里的导入路径相对于您执行脚本的位置是正确的
# 例如，如果 archs 文件夹和 evaluate_psnr.py 在同一目录，就是 from archs.SwinSSC_arch import SwinSSC
from archs.SwinSSC_arch import SwinSSC

def calculate_psnr(img1, img2, max_val=1.0):
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the training YAML config file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model file (.pth).')
    parser.add_argument('--input', type=str, required=True, help='Input test image folder (Ground Truth), e.g., /path/to/kodak24.')
    parser.add_argument('--output', type=str, default='results/SwinSSC_PSNR_Analysis', help='Output folder for the plot and results.')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing.')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Could not load config file {args.config}. Details: {e}")
        return
        
    network_config = config.get('network_g', config) 
    model_args = network_config.get('args', {})
    encoder_kwargs = network_config.get('encoder_kwargs', {})
    decoder_kwargs = network_config.get('decoder_kwargs', {})
    rq_kwargs = network_config.get('rq_kwargs', {})

    model = SwinSSC(
        args=argparse.Namespace(**model_args),
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        rq_kwargs=rq_kwargs
    )
    
    load_net = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    key = 'params_ema' if 'params_ema' in load_net else 'params'
    print(f"Loading model parameters from key: '{key}'")
    model.load_state_dict(load_net[key], strict=True)
    
    model.eval()
    model = model.to(device)
    
    window_size = encoder_kwargs.get('window_size', 8) 
    os.makedirs(args.output, exist_ok=True)

    snr_range = list(range(0, 18, 2))
    psnr_results = {snr: [] for snr in snr_range}

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_files_in_folder = glob.glob(os.path.join(args.input, '*'))
    # 过滤出图像文件，并排序
    image_paths = sorted([p for p in all_files_in_folder if p.lower().endswith(image_extensions)])
    if not image_paths:
        print(f"Error: No images found in the input directory: {args.input}")
        return

    for path in tqdm(image_paths, desc="Processing Images"):
        imgname = os.path.splitext(os.path.basename(path))[0]
        
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_gt = torch.from_numpy(np.transpose(img_gt[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

        _, _, h_old, w_old = img_gt.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_padded = torch.cat([img_gt, torch.flip(img_gt, [2])], 2)[:, :, :h_old + h_pad, :]
        img_padded = torch.cat([img_padded, torch.flip(img_padded, [3])], 3)[:, :, :, :w_old + w_pad]
        
        for snr in snr_range:
            try:
                with torch.no_grad():
                    output_padded = test(img_padded, model, args, window_size, snr)
                    output = output_padded[..., :h_old, :w_old]
                    psnr = calculate_psnr(img_gt, output)
                    psnr_results[snr].append(psnr.item())
            except Exception as error:
                print(f'\nError processing {imgname} at SNR {snr}: {error}')

    avg_psnr_list = []
    print("\n--- Average PSNR Results on Kodak24 ---")
    results_text = "SNR (dB), Average PSNR (dB)\n"
    for snr in snr_range:
        avg_psnr = np.mean(psnr_results[snr]) if psnr_results[snr] else 0
        avg_psnr_list.append(avg_psnr)
        print(f"SNR: {snr:2d} dB  |  Average PSNR: {avg_psnr:.4f} dB")
        results_text += f"{snr}, {avg_psnr:.4f}\n"

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