import numpy as np
import pyldpc
import time
import os
import sys
import subprocess
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import multiprocessing # 1. 导入并行处理库

from faim_bpg import FA_IM_Channel

# import warnings
# from pyldpc.decoder import UserWarning
# # 忽略来自 pyldpc.decoder 的特定 UserWarning
# warnings.filterwarnings("ignore", category=UserWarning, module="pyldpc.decoder")

# ==============================================================================
# 配置参数 (已使用os.path.join实现跨平台)
# ==============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

TARGET_BPP = 1.0
bpp_suffix = f"_bpp{TARGET_BPP}"
n_ldpc = 50
d_v = 3
d_c = 5
MAX_LDPC_ITER = 50

BASE_DIR = './BPG-LDPC' 
DATA_DIR = 'Kodak24'
BPGDEC_EXECUTABLE = os.path.join(BASE_DIR, 'bpgdec.exe')
ROOT_DIR = os.path.join(BASE_DIR, DATA_DIR, 'original_data')
LDPC_ENCODED_DIR = os.path.join(BASE_DIR, DATA_DIR, f'ldpc_encoded{bpp_suffix}')
BPG_ENCODED_DIR = os.path.join(BASE_DIR, DATA_DIR, f'bpg_encoded{bpp_suffix}')
# DECODED_OUTPUT_DIR = os.path.join(BASE_DIR, DATA_DIR, f'decoded_images{bpp_suffix}')
RESULTS_DIR = os.path.join(BASE_DIR, DATA_DIR, f'results{bpp_suffix}')

# --- 模拟参数 ---
SNR_INTERVAL = 2
SNR_START = 18
SNR_END = 20 + SNR_INTERVAL
SNR_range = np.arange(SNR_START, SNR_END, SNR_INTERVAL)

# --- FA-IM 信道参数 ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
FA_IM_K = 4           # 活动端口的数量 (必须是2的幂)
FA_IM_N = 16          # 可用的总端口数量
FA_IM_Nr = 8          # 接收天线的数量
FA_IM_M = 64           # QAM调制的阶数 (4 for QPSK, 16 for 16-QAM)
FA_IM_NUM_H = 2       # 信道实现数量 (信道池大小)
FA_IM_W = 2.0         # 发射端流体天线的总宽度 (米)
FA_IM_L_PATHS = 10    # 多径信道的路径数量

# ==============================================================================
# 辅助函数
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

def execute_command(command_list):
    """安全、跨平台地执行外部命令。"""
    is_windows = sys.platform == "win32"
    if not is_windows:
        command_list.insert(0, "wine")
        # command_list.insert(0, "xvfb-run")

    my_env = os.environ.copy()
    if not is_windows: my_env["WINEDEBUG"] = "fixme-all,err-all"
    try:
        subprocess.run(command_list, env=my_env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {' '.join(command_list)}\n错误信息:\n{e.stderr}")
        raise

# ==============================================================================
# 2. 并行任务函数 (封装处理单个图片在所有SNR下的逻辑)
# ==============================================================================
def process_image_all_snrs(mat_filename):
    """
    处理单个编码文件在所有SNR下的解码和评估。
    返回一个字典，包含这张图片在每个SNR下的PSNR和MS-SSIM值。
    """
    # 子进程将继承全局变量 H, G, k_ldpc, n_actual_ldpc, channel, ms_ssim_window, ms_ssim_weights
    pid = os.getpid()
    image_base_name, _ = os.path.splitext(mat_filename)
    
    # 初始化此图片的结果容器
    image_psnr_results = {snr: [] for snr in SNR_range}
    image_ms_ssim_results = {snr: [] for snr in SNR_range}

    try:
        # 加载数据
        original_image_path = os.path.join(ROOT_DIR, image_base_name + '.png')
        original_img = Image.open(original_image_path).convert('RGB')
        original_tensor = to_tensor(original_img).unsqueeze(0).to(DEVICE)

        mat_path = os.path.join(LDPC_ENCODED_DIR, mat_filename)
        encoded_data = loadmat(mat_path)
        encoded_bits_tensor = torch.from_numpy(encoded_data['encoded_bits'].flatten())

        bpg_bin_path = os.path.join(BPG_ENCODED_DIR, image_base_name + '.bin')
        original_bpg_bit_length = os.path.getsize(bpg_bin_path) * 8

        # 遍历所有SNR
        for snr in SNR_range:
            print(f"--- [PID:{pid}] Processing: {image_base_name} at SNR={snr} ---")
            psnr_per_channel = []
            ms_ssim_per_channel = []
            for idx_H in range(FA_IM_NUM_H):
                # with torch.no_grad():
                #     llr_tensor = channel(encoded_bits_tensor, snr_db=snr, idx_H=idx_H)
                # llr_numpy = llr_tensor.cpu().numpy()
                llr_numpy = 1 - 2 * encoded_bits_tensor.numpy()

                num_blocks = len(llr_numpy) // n_actual_ldpc
                llr_blocks = llr_numpy.reshape(num_blocks, n_actual_ldpc)

                y_blocks = np.vstack([pyldpc.decode(H, llr_blocks[i], snr=100, maxiter=MAX_LDPC_ITER) for i in range(num_blocks)])
                decoded_message_stream = np.concatenate([pyldpc.get_message(G, y_blocks[i]) for i in range(num_blocks)])


                # # ==================== DEBUG PRINT 3 ====================
                # # 从 bpg_encoded 目录读取原始未编码的比特流
                # with open(bpg_bin_path, 'rb') as f:
                #     original_info_bits = np.unpackbits(np.fromfile(f, dtype=np.uint8))

                # # 对比解码后的信息比特流和原始信息比特流
                # comparison_len = min(len(original_info_bits), len(decoded_message_stream))
                # errors = np.sum(original_info_bits[:comparison_len] != decoded_message_stream[:comparison_len])
                # ber = errors / comparison_len

                # print(f"\n[DEBUG 3] Post-Decoding Verification")
                # print(f"  - Decoded message stream length: {len(decoded_message_stream)}")
                # print(f"  - Original info bits length:   {len(original_info_bits)}")
                # print(f"  - Bit Errors (BER): {errors}/{comparison_len} = {ber:.6f}")
                # # =======================================================

                remainder = original_bpg_bit_length % k_ldpc
                padding_len = (k_ldpc - remainder) % k_ldpc
                decoded_bpg_bits = decoded_message_stream[:-padding_len] if padding_len > 0 else decoded_message_stream
                
                assert len(decoded_bpg_bits) == original_bpg_bit_length
                
                # BPG解码
                decoded_bytes = np.packbits(decoded_bpg_bits.astype(np.uint8))
                # 创建唯一的临时文件名
                temp_bin_path = os.path.join(RESULTS_DIR, f"{image_base_name}_temp_{pid}.bin")
                with open(temp_bin_path, 'wb') as f: 
                    decoded_bytes.tofile(f)
                
                decoded_image_path = os.path.join(RESULTS_DIR, f"{image_base_name}_snr{snr}_H{idx_H}.png")
                try:
                    execute_command([BPGDEC_EXECUTABLE, '-o', decoded_image_path, temp_bin_path])
                    os.remove(temp_bin_path)
                except subprocess.CalledProcessError:
                    # 如果 execute_command 失败 (返回非零)，捕获异常
                    print(f"    -> [PID:{pid}] BPG decoding FAILED for {image_base_name} at SNR={snr} dB. Skipping metrics for this channel realization.")
                    # 清理失败的临时文件
                    if os.path.exists(temp_bin_path):
                        os.remove(temp_bin_path)
                    # 跳过这次信道实现的后续步骤，继续下一个
                    continue 
                

                if not os.path.exists(decoded_image_path): continue

                # 计算指标
                decoded_img = Image.open(decoded_image_path).convert('RGB')
                decoded_tensor = to_tensor(decoded_img).unsqueeze(0).to(DEVICE)
                
                psnr_per_channel.append(calculate_psnr(original_tensor, decoded_tensor, 1.0).item())
                ms_ssim_per_channel.append(calculate_ms_ssim(original_tensor, decoded_tensor, ms_ssim_window, 1.0, ms_ssim_weights).item())
                os.remove(decoded_image_path) # 清理中间图片文件

            if psnr_per_channel: # 仅当成功解码时才记录
                image_psnr_results[snr] = np.mean(psnr_per_channel)
                image_ms_ssim_results[snr] = np.mean(ms_ssim_per_channel)

        print(f"--- [PID:{pid}] Finished: {image_base_name} ---")
        return (image_psnr_results, image_ms_ssim_results)

    except Exception as e:
        print(f"--- [PID:{pid}] FAILED to process {image_base_name}: {e} ---")
        return None # 返回None表示失败

def init_worker(h_matrix, g_matrix, msssim_win, msssim_w):
    """
    这个函数会在每个子进程启动时被调用一次。
    它的作用是接收父进程传递过来的数据，并创建子进程专属的全局对象。
    """

    # 将接收到的数据存储为子进程的全局变量
    # 这样，process_image_all_snrs 函数就能直接访问它们了
    global H, G, k_ldpc, n_actual_ldpc, channel, ms_ssim_window, ms_ssim_weights

    # --- 初始化LDPC矩阵 ---
    H = h_matrix
    G = g_matrix
    k_ldpc = G.shape[1]
    n_actual_ldpc = G.shape[0]

    # --- 初始化FA-IM信道 ---
    channel = FA_IM_Channel(
        K=FA_IM_K, N=FA_IM_N, Nr=FA_IM_Nr, M=FA_IM_M,
        num_H=FA_IM_NUM_H, W=FA_IM_W, L_paths=FA_IM_L_PATHS, device=DEVICE
    )
    
    # --- 初始化MS-SSIM组件 ---
    ms_ssim_window = msssim_win
    ms_ssim_weights = msssim_w

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # 如果已经设置过了，可能会抛出 RuntimeError，可以安全地忽略
        pass

    # 在主进程中初始化一次，子进程将继承这些变量
    matrix_load_path = os.path.join(LDPC_ENCODED_DIR, 'ldpc_matrices.npz')
    if not os.path.exists(matrix_load_path):
        raise FileNotFoundError(f"LDPC matrix file not found at {matrix_load_path}.")
    matrices = np.load(matrix_load_path)
    h_main, g_main = matrices['H'].astype(np.int32), matrices['G'].astype(np.int32)

    # H, G = pyldpc.make_ldpc(n_ldpc, d_v, d_c, systematic=True, sparse=True)
    h_main, g_main = pyldpc.make_ldpc(n_ldpc, d_v, d_c, systematic=True, sparse=True)

    msssim_win_main = create_window(11, 3).to(DEVICE)
    msssim_w_main = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=DEVICE)

    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- 准备文件列表 ---
    mat_files = sorted([f for f in os.listdir(LDPC_ENCODED_DIR) if f.lower().endswith('.mat')])
    
    # --- 3. 使用 multiprocessing.Pool 进行并行处理 ---
    num_processes = max(1, multiprocessing.cpu_count() - 10)
    
    start_time = time.time()

    init_args = (h_main, g_main, msssim_win_main, msssim_w_main)
    with multiprocessing.Pool(processes=num_processes,
                              initializer=init_worker,
                              initargs=init_args) as pool:
        all_results = pool.map(process_image_all_snrs, mat_files)
    
    # 单线程测试
    # mat_files = sorted([f for f in os.listdir(LDPC_ENCODED_DIR) if f.lower().endswith('.mat')])[:1]
    # init_worker(h_main, g_main, msssim_win_main, msssim_w_main)
    # all_results = [process_image_all_snrs(mat_files[0])]
    
    # --- 4. 聚合所有图片的结果 ---
    # 过滤掉失败的任务 (返回None的)
    successful_results = [res for res in all_results if res is not None]
    
    # 初始化总结果容器
    total_psnr_results = {snr: [] for snr in SNR_range}
    total_ms_ssim_results = {snr: [] for snr in SNR_range}

    for psnr_res, msssim_res in successful_results:
        for snr in SNR_range:
            if psnr_res[snr]: # 如果列表不为空
                total_psnr_results[snr].append(psnr_res[snr])
            if msssim_res[snr]:
                total_ms_ssim_results[snr].append(msssim_res[snr])

    # --- 汇总、保存和绘制结果 ---
    avg_psnr_list, avg_ms_ssim_list = [], []
    print("\n--- Average Performance on Kodak24 Dataset ---")
    results_text = "SNR (dB),Avg PSNR (dB),Avg MS-SSIM\n"
    for snr in SNR_range:
        avg_psnr = np.mean(total_psnr_results[snr]) if total_psnr_results[snr] else 0
        avg_ms_ssim = np.mean(total_ms_ssim_results[snr]) if total_ms_ssim_results[snr] else 0
        avg_psnr_list.append(avg_psnr)
        avg_ms_ssim_list.append(avg_ms_ssim)
        print(f"SNR: {snr:2d} dB | Avg PSNR: {avg_psnr:.4f} dB | Avg MS-SSIM: {avg_ms_ssim:.4f}")
        results_text += f"{snr},{avg_psnr:.4f},{avg_ms_ssim:.4f}\n"

    # 保存数值结果到 CSV 文件
    results_file_path = os.path.join(RESULTS_DIR, 'results.csv')
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
    plot_path_psnr = os.path.join(RESULTS_DIR, 'PSNR_vs_SNR_curve.png')
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
    plot_path_msssim = os.path.join(RESULTS_DIR, 'MS_SSIM_vs_SNR_curve.png')
    plt.savefig(plot_path_msssim)
    print(f"MS-SSIM plot saved to {plot_path_msssim}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")