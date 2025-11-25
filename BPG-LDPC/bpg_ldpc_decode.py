import numpy as np
import pyldpc
import time
import os
import sys
import subprocess
from scipy.io import loadmat
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import pandas as pd

from faim_bpg import FA_IM_Channel

# import warnings
# from pyldpc.decoder import UserWarning
# # 忽略来自 pyldpc.decoder 的特定 UserWarning
# warnings.filterwarnings("ignore", category=UserWarning, module="pyldpc.decoder")

# use fork
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
torch.set_num_threads(1)


# ==============================================================================
# 配置参数
# ==============================================================================
TARGET_BPP = 3.0
bpp_suffix = f"_bpp{TARGET_BPP}"
n_ldpc = 1944
d_v = 23
d_c = 24
# n_ldpc = 1296
# d_v = 4
# d_c = 6
MAX_LDPC_ITER = 20

BASE_DIR = './BPG-LDPC' 
DATA_DIR = 'Kodak24'
BPGDEC_EXECUTABLE = os.path.join(BASE_DIR, 'bpgdec.exe')
ROOT_DIR = os.path.join(BASE_DIR, DATA_DIR, 'data_resize')
LDPC_ENCODED_DIR = os.path.join(BASE_DIR, DATA_DIR, f'encode_resize_{n_ldpc}_{d_v}_{d_c}', f'ldpc_encoded{bpp_suffix}')
BPG_ENCODED_DIR = os.path.join(BASE_DIR, DATA_DIR, f'encode_resize_{n_ldpc}_{d_v}_{d_c}', f'bpg_encoded{bpp_suffix}')
# DECODED_OUTPUT_DIR = os.path.join(BASE_DIR, DATA_DIR, f'decoded_images{bpp_suffix}')
RESULTS_DIR = os.path.join(BASE_DIR, DATA_DIR, f'results_resize_{n_ldpc}_{d_v}_{d_c}', f'results{bpp_suffix}')

# --- 模拟参数 ---
SNR_START = 0
SNR_INTERVAL = 2
SNR_END = 24 + SNR_INTERVAL
SNR_range = np.arange(SNR_START, SNR_END, SNR_INTERVAL)

TRANS_NUM = 10  # 每个SNR点的传输次数
# --- FA-IM 信道参数 ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
FA_IM_K = 4           # 活动端口的数量 (必须是2的幂)
FA_IM_N = 16          # 可用的总端口数量
FA_IM_Nr = 8          # 接收天线的数量
FA_IM_M = 64           # QAM调制的阶数 (4 for QPSK, 16 for 16-QAM)
FA_IM_NUM_H = 500       # 信道实现数量 (信道池大小)

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
        tqdm.write(f"命令执行失败: {' '.join(command_list)}\n错误信息:\n{e.stderr}")
        raise

# ==============================================================================
# 2. 并行任务函数
# ==============================================================================
def process_single_task(mat_filename, snr, idx_H):
    """
    处理单个解码任务：一张图片、一个SNR、一次传输(idx_H)。
    返回一个元组: (snr, psnr_value, msssim_value)。
    """
    # 子进程会从 init_worker 继承全局变量 H, G, k_ldpc, n_actual_ldpc, channel 等
    pid = os.getpid()
    image_base_name, _ = os.path.splitext(mat_filename)
    
    tqdm.write(f"--- [PID:{pid}] Task Start: {image_base_name} | SNR={snr} | H_idx={idx_H} ---")

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

        with torch.no_grad():
            llr_tensor = channel(encoded_bits_tensor, snr_db=snr)
        llr_numpy = llr_tensor.cpu().to(torch.float64).numpy()
        # llr_numpy = 1 - 2 * encoded_bits_tensor.numpy()

        num_blocks = len(llr_numpy) // n_actual_ldpc
        llr_blocks = llr_numpy.reshape(num_blocks, n_actual_ldpc)

        y_blocks = np.vstack([pyldpc.decode(H, llr_blocks[i], snr=snr, maxiter=MAX_LDPC_ITER) for i in range(num_blocks)])
        decoded_message_stream = np.concatenate([pyldpc.get_message(G, y_blocks[i]) for i in range(num_blocks)])

        remainder = original_bpg_bit_length % k_ldpc
        padding_len = (k_ldpc - remainder) % k_ldpc
        decoded_bpg_bits = decoded_message_stream[:-padding_len] if padding_len > 0 else decoded_message_stream
        
        assert len(decoded_bpg_bits) == original_bpg_bit_length
        
        # BPG解码
        decoded_bytes = np.packbits(decoded_bpg_bits.astype(np.uint8))
        temp_bin_path = os.path.join(RESULTS_DIR, f"{image_base_name}_temp_{pid}_{snr}_{idx_H}.bin")
        with open(temp_bin_path, 'wb') as f:
            decoded_bytes.tofile(f)
        
        decoded_image_path = os.path.join(RESULTS_DIR, f"{image_base_name}_snr{snr}_H{idx_H}.png")
        
        psnr_val, msssim_val = None, None
        try:
            execute_command([BPGDEC_EXECUTABLE, '-o', decoded_image_path, temp_bin_path])
            if os.path.exists(decoded_image_path):
                decoded_img = Image.open(decoded_image_path).convert('RGB')
                decoded_tensor = to_tensor(decoded_img).unsqueeze(0).to(DEVICE)
                
                psnr_val = calculate_psnr(original_tensor, decoded_tensor, 1.0).item()
                msssim_val = calculate_ms_ssim(original_tensor, decoded_tensor, ms_ssim_window, 1.0, ms_ssim_weights).item()
        
        except subprocess.CalledProcessError:
            tqdm.write(f"    -> [PID:{pid}] BPG decoding FAILED for task. Skipping metrics.")
        finally:
             if os.path.exists(temp_bin_path): os.remove(temp_bin_path)
             if os.path.exists(decoded_image_path): os.remove(decoded_image_path)

        # 无论成功失败，都写入一行，失败时指标为空
        psnr_str = f"{psnr_val:.4f}" if psnr_val is not None else ""
        msssim_str = f"{msssim_val:.4f}" if msssim_val is not None else ""
        result_line = f"{image_base_name},{snr},{idx_H},{psnr_str},{msssim_str}\n"

        # 使用从父进程继承来的锁来保护文件写入操作
        LOCK.acquire()
        try:
            with open(RESULT_FILE_PATH, 'a') as f:
                f.write(result_line)
        finally:
            LOCK.release() # 确保锁总是被释放
        
        if psnr_val is not None:
             tqdm.write(f"--- [PID:{pid}] Task OK:    {image_base_name} | SNR={snr} | H_idx={idx_H} | PSNR={psnr_val:.4f} ---")
        # -----------------------------------------------------

        return "OK" # 只返回一个简单的状态字符串

    except Exception as e:
        tqdm.write(f"--- [PID:{pid}] Task FAILED: {image_base_name} | SNR={snr} | H_idx={idx_H} with error: {e} ---")
        result_line = f"{image_base_name},{snr},{idx_H},,FAIL: {e}\n"
        LOCK.acquire()
        try:
            with open(RESULT_FILE_PATH, 'a') as f:
                f.write(result_line)
        finally:
            LOCK.release()
        return "FAIL"


def init_worker(h_matrix, g_matrix, msssim_win, msssim_w, channel_obj, result_filepath, lock):
    """
    这个函数会在每个子进程启动时被调用一次。
    它的作用是接收父进程传递过来的数据，并创建子进程专属的全局对象。
    """

    # 将接收到的数据存储为子进程的全局变量
    # 这样，process_image_all_snrs 函数就能直接访问它们了
    global H, G, k_ldpc, n_actual_ldpc, channel, ms_ssim_window, ms_ssim_weights

    global RESULT_FILE_PATH, LOCK

    # --- 初始化LDPC矩阵 ---
    H = h_matrix
    G = g_matrix
    k_ldpc = G.shape[1]
    n_actual_ldpc = G.shape[0]

    # # --- 初始化FA-IM信道 ---
    # channel = FA_IM_Channel(
    #     K=FA_IM_K, N=FA_IM_N, Nr=FA_IM_Nr, M=FA_IM_M,
    #     num_H=FA_IM_NUM_H, W=FA_IM_W, L_paths=FA_IM_L_PATHS, device=DEVICE
    # )

    channel = channel_obj
    
    # --- 初始化MS-SSIM组件 ---
    ms_ssim_window = msssim_win
    ms_ssim_weights = msssim_w

    RESULT_FILE_PATH = result_filepath
    LOCK = lock


# 创建一个辅助函数，用于解包参数
# 它的作用就是接收一个元组 (arg1, arg2, ...)，然后用 *args 的形式传给你的真实任务函数
def starmap_helper(args):
    """
    辅助函数，用于 imap 系列函数调用需要解包参数的函数
    """
    return process_single_task(*args)
# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # use gpu
    # try:
    #     multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     # 如果已经设置过了，可能会抛出 RuntimeError，可以安全地忽略
    #     pass

    # 在主进程中初始化一次，子进程将继承这些变量
    matrix_load_path = os.path.join(LDPC_ENCODED_DIR, 'ldpc_matrices.npz')
    if not os.path.exists(matrix_load_path):
        raise FileNotFoundError(f"LDPC matrix file not found at {matrix_load_path}.")
    matrices = np.load(matrix_load_path)
    h_main, g_main = matrices['H'], matrices['G']

    # h_main, g_main = pyldpc.make_ldpc(n_ldpc, d_v, d_c, systematic=True, sparse=True)

    msssim_win_main = create_window(11, 3).to(DEVICE)
    msssim_w_main = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=DEVICE)

    print("Creating FA-IM Channel...")
    channel_main = FA_IM_Channel(
        K=FA_IM_K, N=FA_IM_N, Nr=FA_IM_Nr, M=FA_IM_M,
        num_H=FA_IM_NUM_H, W=FA_IM_W, L_paths=FA_IM_L_PATHS, device=DEVICE
    )
    print(f"FA-IM Channel initialization complete.")

    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- 准备文件列表 ---
    mat_files = sorted([f for f in os.listdir(LDPC_ENCODED_DIR) if f.lower().endswith('.mat')])
    tasks = []
    for filename in mat_files:
        for snr_val in SNR_range:
            for h_idx in range(TRANS_NUM):
                tasks.append((filename, snr_val, h_idx))

    # --- 使用 multiprocessing.Pool 进行并行处理 ---
    num_processes = min(len(tasks), max(1, multiprocessing.cpu_count() - 10))
    
    start_time = time.time()

    results_csv_path = os.path.join(RESULTS_DIR, 'live_results.csv')
    # 在开始前创建结果文件并写入表头
    with open(results_csv_path, 'w') as f:
        f.write("image_name,snr,h_idx,psnr,ms_ssim\n")

    # 创建一个进程安全的锁
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # 将文件路径和锁添加到 init_args
    init_args = (h_main, g_main, msssim_win_main, msssim_w_main, channel_main, results_csv_path, lock)
    # ------------------------------------------------------------

    # --- 核心修改: 使用 maxtasksperchild 并只迭代不收集结果 ---
    with multiprocessing.Pool(processes=num_processes,
                              initializer=init_worker,
                              initargs=init_args,
                              maxtasksperchild=100) as pool: # 4. 添加 maxtasksperchild
        
        results_iterator = pool.imap_unordered(starmap_helper, tasks)
        
        # 只迭代以驱动进度条，不再用 list() 收集结果
        for _ in tqdm(results_iterator, total=len(tasks), desc="并发任务进度"):
            pass
    
    # 单线程测试
    # init_worker(h_main, g_main, msssim_win_main, msssim_w_main)
    # all_results = [process_single_task(*tasks[0])]
    
    try:
        results_df = pd.read_csv(results_csv_path)
    except pd.errors.EmptyDataError:
        print("错误：结果文件为空，无法进行汇总。")
        sys.exit(1)
    
    # 删除psnr列为空的行，这些是解码失败的任务
    results_df.dropna(subset=['psnr'], inplace=True)
    # 确保数据类型正确
    results_df['snr'] = results_df['snr'].astype(int)
    results_df['psnr'] = results_df['psnr'].astype(float)
    results_df['ms_ssim'] = results_df['ms_ssim'].astype(float)

    # 按SNR分组并计算平均值
    summary = results_df.groupby('snr')[['psnr', 'ms_ssim']].mean().reset_index()
    summary.rename(columns={'psnr': 'Avg PSNR (dB)', 'ms_ssim': 'Avg MS-SSIM'}, inplace=True)
    
    avg_psnr_list = summary['Avg PSNR (dB)'].tolist()
    avg_ms_ssim_list = summary['Avg MS-SSIM'].tolist()

    # 打印汇总结果
    print("\n--- Average Performance on Kodak24 Dataset ---")
    for index, row in summary.iterrows():
        print(f"SNR: {int(row['snr']):2d} dB | Avg PSNR: {row['Avg PSNR (dB)']:.4f} dB | Avg MS-SSIM: {row['Avg MS-SSIM']:.4f}")
    
    # 保存汇总结果
    summary_file_path = os.path.join(RESULTS_DIR, 'results.csv')
    summary.to_csv(summary_file_path, index=False)
    print(f"\nSummary results saved to {summary_file_path}")

    # 绘制并保存PSNR曲线
    plt.figure(figsize=(10, 7))
    plt.plot(summary['snr'], avg_psnr_list, marker='o', linestyle='-', label=f'BPG+LDPC (FA-IM, BPP={TARGET_BPP})')
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
    plt.plot(summary['snr'], avg_ms_ssim_list, marker='s', linestyle='--', color='crimson', label=f'BPG+LDPC (FA-IM, BPP={TARGET_BPP})')
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