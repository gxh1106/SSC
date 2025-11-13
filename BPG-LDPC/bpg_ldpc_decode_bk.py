import numpy as np
import pyldpc
import time
import os
import subprocess
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

from faim_bpg import FA_IM_Channel

# ==============================================================================
# 配置参数 (与编码器脚本匹配)
# ==============================================================================
# --- 目标码率 ---
TARGET_BPP = 1.0
bpp_suffix = f"_bpp{TARGET_BPP}"
# --- LDPC 参数 ---
n_ldpc = 50   # LDPC 码长 (Codeword length)
d_v = 3       # 变量节点度 (Variable node degree)
d_c = 5       # 校验节点度 (Check node degree)
MAX_LDPC_ITER = 50 # LDPC解码器最大迭代次数

# --- BPG 可执行文件路径 ---
BPGDEC_PATH = '.\\bpgdec' # 根据您的bpgdec.exe位置进行修改
# --- 路径设置 ---
root_dir = '.\\Kodak24\\original_data\\'                          # 原始图片路径 (用于对比)
ldpc_encoded_dir = f'.\\Kodak24\\ldpc_encoded{bpp_suffix}\\'      # 存储.mat文件的路径
bpg_encoded_dir = f'.\\Kodak24\\bpg_encoded{bpp_suffix}\\'       # BPG编码结果 (用于获取原始长度)
decoded_output_dir = f'.\\Kodak24\\decoded_images{bpp_suffix}\\'  # 存储解码后图片的路径
results_dir = f'.\\Kodak24\\results{bpp_suffix}\\' # 存储CSV和图表的新目录

# --- 模拟参数 ---
# 要测试的信噪比范围 (dB)
SNR_INTERVAL = 2
SNR_START = 0
SNR_END = 20 + SNR_INTERVAL
SNR_range = np.arange(SNR_START, SNR_END, SNR_INTERVAL)

# --- FA-IM 信道参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FA_IM_K = 4           # 活动端口的数量 (必须是2的幂)
FA_IM_N = 16          # 可用的总端口数量
FA_IM_Nr = 8          # 接收天线的数量
FA_IM_M = 64           # QAM调制的阶数 (4 for QPSK, 16 for 16-QAM)
FA_IM_NUM_H = 2       # 信道实现数量 (信道池大小)
FA_IM_W = 2.0         # 发射端流体天线的总宽度 (米)
FA_IM_L_PATHS = 10    # 多径信道的路径数量

# ==============================================================================
# 提供的图像质量评估函数
# ==============================================================================
def calculate_psnr(img1, img2, max_val=1.0):
    """计算两张图的PSNR值"""
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
    """计算 MS-SSIM"""
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


# ==============================================================================
# 主程序
# ==============================================================================

# --- 初始化LDPC码 (必须与编码器一致) ---
# H, G = pyldpc.make_ldpc(n_ldpc, d_v, d_c, systematic=True, sparse=True)
matrix_load_path = f'.\\Kodak24\\ldpc_encoded{bpp_suffix}\\ldpc_matrices.npz'
if not os.path.exists(matrix_load_path):
    raise FileNotFoundError(f"LDPC matrix file not found at {matrix_load_path}. Please run the encoder first.")
matrices = np.load(matrix_load_path)
H = matrices['H'].astype(np.int32)
G = matrices['G'].astype(np.int32)

k_ldpc = G.shape[1]
n_actual_ldpc = G.shape[0]



# --- 初始化 MS-SSIM 所需的组件 ---
ms_ssim_window = create_window(11, 3).to(DEVICE)
ms_ssim_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=DEVICE)

# --- 创建输出目录 ---
if not os.path.exists(decoded_output_dir):
    os.makedirs(decoded_output_dir)
os.makedirs(results_dir, exist_ok=True)

# --- 初始化信道 ---
channel = FA_IM_Channel(
    K=FA_IM_K, N=FA_IM_N, Nr=FA_IM_Nr, M=FA_IM_M,
    num_H=FA_IM_NUM_H, W=FA_IM_W, L_paths=FA_IM_L_PATHS, device=DEVICE
)


psnr_results = {snr: [] for snr in SNR_range}
ms_ssim_results = {snr: [] for snr in SNR_range}

mat_files = [f for f in os.listdir(ldpc_encoded_dir) if f.lower().endswith('.mat')]
total_images = len(mat_files)

# --- 遍历所有编码文件进行解码和评估 ---
for idx, mat_filename in enumerate(mat_files):
    image_base_name = mat_filename.split('.')[0]
    print(f"\n[{idx+1}/{total_images}] Processing: {image_base_name}")

    # 加载原始图像和编码数据
    original_image_path = os.path.join(root_dir, image_base_name + '.png')
    original_img = Image.open(original_image_path).convert('RGB')
    original_tensor = to_tensor(original_img).unsqueeze(0).to(DEVICE)

    mat_path = os.path.join(ldpc_encoded_dir, mat_filename)
    encoded_data = loadmat(mat_path)
    encoded_bits_tensor = torch.from_numpy(encoded_data['encoded_bits'].flatten())

    bpg_bin_path = os.path.join(bpg_encoded_dir, image_base_name + '.bin')
    original_bpg_bit_length = os.path.getsize(bpg_bin_path) * 8

    # 针对不同SNR进行处理
    for snr in SNR_range:
        psnr_per_channel = []
        ms_ssim_per_channel = []
        for idx_H in range(FA_IM_NUM_H):
            # 通过信道并解码
            with torch.no_grad():
                llr_tensor = channel(encoded_bits_tensor, snr_db = snr, idx_H=idx_H)
            llr_numpy = llr_tensor.cpu().numpy()

            # llr_numpy = 1 - 2 * encoded_bits_tensor.numpy()

            # 将 LLRs 重塑为 (num_blocks, n_ldpc) 的形状
            num_blocks = len(llr_numpy) // n_actual_ldpc
            llr_blocks = llr_numpy.reshape(num_blocks, n_actual_ldpc)

            # 1. 逐块解码，得到估计的码字 y
            # pyldpc.decode 返回 (message, codeword)
            y_blocks = np.vstack([pyldpc.decode(H, llr_blocks[i], snr=0, maxiter=MAX_LDPC_ITER) for i in range(num_blocks)])
            # 2. 从每个码字中提取消息比特
            decoded_message_stream = np.concatenate([pyldpc.get_message(G, y_blocks[i]) for i in range(num_blocks)])

            # 3. 计算并移除填充
            remainder = original_bpg_bit_length % k_ldpc
            padding_len = (k_ldpc - remainder) % k_ldpc # 如果 remainder 是 0, padding_len 也是 0
            
            if padding_len > 0:
                decoded_bpg_bits = decoded_message_stream[:-padding_len]
            else:
                decoded_bpg_bits = decoded_message_stream
            
            # 断言确保长度正确
            assert len(decoded_bpg_bits) == original_bpg_bit_length
            
            # BPG解码

            decoded_bytes = np.packbits(decoded_bpg_bits)
            temp_bin_path = os.path.join(decoded_output_dir, f"{image_base_name}_temp.bin")
            with open(temp_bin_path, 'wb') as f:
                decoded_bytes.tofile(f)

            decoded_image_path = os.path.join(decoded_output_dir, f"{image_base_name}.png")
            bpg_decode_command = f'{BPGDEC_PATH} -o "{decoded_image_path}" "{temp_bin_path}"'
            subprocess.run(bpg_decode_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            os.remove(temp_bin_path)
            
            if not os.path.exists(decoded_image_path):
                print(" -> BPG decoding FAILED. Skipping metrics.")
                continue

            # 计算并存储指标
            decoded_img = Image.open(decoded_image_path).convert('RGB')
            decoded_tensor = to_tensor(decoded_img).unsqueeze(0).to(DEVICE)
            
            psnr_val = calculate_psnr(original_tensor, decoded_tensor, max_val=1.0)
            ms_ssim_val = calculate_ms_ssim(original_tensor, decoded_tensor, window=ms_ssim_window, data_range=1.0, weights=ms_ssim_weights)
            psnr_per_channel.append(psnr_val.item())
            ms_ssim_per_channel.append(ms_ssim_val.item())

        psnr_results[snr].append(np.mean(psnr_per_channel))
        ms_ssim_results[snr].append(np.mean(ms_ssim_per_channel))
        # print(f" -> PSNR: {psnr_val.item():.2f}, MS-SSIM: {ms_ssim_val.item():.4f}")


# --- 汇总、保存和绘制结果 ---
avg_psnr_list = []
avg_ms_ssim_list = []
print("\n--- Average Performance on Kodak24 Dataset ---")
results_text = "SNR (dB), Avg PSNR (dB), Avg MS-SSIM\n"

for snr in SNR_range:
    avg_psnr = np.mean(psnr_results[snr]) if psnr_results[snr] else 0
    avg_ms_ssim = np.mean(ms_ssim_results[snr]) if ms_ssim_results[snr] else 0
    avg_psnr_list.append(avg_psnr)
    avg_ms_ssim_list.append(avg_ms_ssim)
    
    print(f"SNR: {snr:2d} dB  |  Avg PSNR: {avg_psnr:.4f} dB  |  Avg MS-SSIM: {avg_ms_ssim:.4f}")
    results_text += f"{snr},{avg_psnr:.4f},{avg_ms_ssim:.4f}\n"

# 保存数值结果到 CSV 文件
results_file_path = os.path.join(results_dir, 'results.csv')
with open(results_file_path, 'w') as f:
    f.write(results_text)
print(f"\nNumeric results saved to {results_file_path}")

# 绘制并保存PSNR曲线
plt.figure(figsize=(10, 7))
plt.plot(SNR_range, avg_psnr_list, marker='o', linestyle='-', label=f'BPG+LDPC (FA-IM, BPP={TARGET_BPP})')
plt.title('PSNR vs. SNR Performance on Kodak24 Dataset')
plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
plt.ylabel('Average Peak Signal-to-Noise Ratio (PSNR) [dB]')
plt.xticks(SNR_range)
plt.grid(True, which='both', linestyle='--')
plt.legend()
plot_path_psnr = os.path.join(results_dir, 'PSNR_vs_SNR_curve.png')
plt.savefig(plot_path_psnr)
print(f"PSNR plot saved to {plot_path_psnr}")

# 绘制并保存MS-SSIM曲线
plt.figure(figsize=(10, 7))
plt.plot(SNR_range, avg_ms_ssim_list, marker='s', linestyle='--', color='crimson', label=f'BPG+LDPC (FA-IM, BPP={TARGET_BPP})')
plt.title('MS-SSIM vs. SNR Performance on Kodak24 Dataset')
plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
plt.ylabel('Average MS-SSIM')
plt.xticks(SNR_range)
plt.grid(True, which='both', linestyle='--')
plt.legend()
plot_path_msssim = os.path.join(results_dir, 'MS_SSIM_vs_SNR_curve.png')
plt.savefig(plot_path_msssim)
print(f"MS-SSIM plot saved to {plot_path_msssim}")

print("\nAnalysis complete.")
