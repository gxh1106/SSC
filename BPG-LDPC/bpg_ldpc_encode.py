import numpy as np
import pyldpc
import time
import os
import sys
import subprocess
from scipy.io import savemat
from PIL import Image
import multiprocessing # 1. 导入并行处理库

# ==============================================================================
# 配置参数
# ==============================================================================
# --- 目标码率 ---
TARGET_BPP = 6.0
bpp_suffix = f"_bpp{TARGET_BPP}"
# --- LDPC 参数 ---
n_ldpc = 50
d_v = 3
d_c = 5
# --- 路径设置 ---
BASE_DIR = './BPG-LDPC'
DATA_DIR = 'Kodak24'
BPGENC_EXECUTABLE = os.path.join(BASE_DIR, 'bpgenc.exe')
ROOT_DIR = os.path.join(BASE_DIR, DATA_DIR, 'original_data')
SAVE_DIR_BPG = os.path.join(BASE_DIR, DATA_DIR, f'bpg_encoded{bpp_suffix}')
SAVE_DIR_LDPC = os.path.join(BASE_DIR, DATA_DIR, f'ldpc_encoded{bpp_suffix}')

# ==============================================================================
# 辅助函数
# ==============================================================================

def binaryproduct(X, Y):
    """在 Z/2Z 中计算矩阵-向量乘积"""
    A = X.dot(Y)
    try: A = A.toarray()
    except AttributeError: pass
    return A % 2

def execute_command(command_list):
    """
    安全、跨平台地执行外部命令。
    在Linux上自动添加 'xvfb-run wine'。
    """
    is_windows = sys.platform == "win32"
    if not is_windows:
        command_list.insert(0, "wine")
        # command_list.insert(0, "xvfb-run")

    my_env = os.environ.copy()
    if not is_windows:
        my_env["WINEDEBUG"] = "fixme-all,err-all"

    try:
        subprocess.run(
            command_list,
            env=my_env,
            check=True,
            capture_output=True, # 隐藏子进程的输出
            text=True
        )
    except subprocess.CalledProcessError as e:
        # 如果命令失败，打印错误信息，这对于调试至关重要
        print(f"命令执行失败: {' '.join(command_list)}")
        print(f"错误信息:\n{e.stderr}")
        raise # 重新抛出异常，让主程序知道任务失败

def find_optimal_qp(image_path, target_bits, bpg_executable, max_iter=6):
    """
    使用二分查找寻找最佳QP值。
    为避免并行冲突，使用进程ID创建唯一的临时文件。
    """
    pid = os.getpid() # 获取当前进程ID
    print(f"--- [PID:{pid}] Find QP for {os.path.basename(image_path)} ---")
    # print(f"    Target bits: {target_bits:,.0f}")

    best_qp, qp_low, qp_high = 51, 0, 51
    min_bits_diff = float('inf')
    
    # 2. 为每个进程创建唯一的临时文件
    temp_output_path = f"temp_search_{pid}.bpg"

    for i in range(max_iter):
        if qp_low > qp_high: break
        
        current_qp = (qp_low + qp_high) // 2
        
        command = [
            bpg_executable, '-q', str(current_qp), '-m', '9', '-b', '8',
            image_path, '-o', temp_output_path
        ]
        try:
            execute_command(command)
        except (subprocess.CalledProcessError, FileNotFoundError):
             # 编码失败，可能QP值无效，向更高QP搜索
            qp_low = current_qp + 1
            continue

        actual_bits = os.path.getsize(temp_output_path) * 8
        diff = abs(actual_bits - target_bits)

        if diff < min_bits_diff:
            min_bits_diff = diff
            best_qp = current_qp

        if actual_bits > target_bits: qp_low = current_qp + 1
        elif actual_bits < target_bits: qp_high = current_qp - 1
        else: break
            
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    # print(f"--- [PID:{pid}] Complete search. Best QP = {best_qp} ---")
    return best_qp


# ==============================================================================
# 3. 并行任务函数 (封装处理单个图片的逻辑)
# ==============================================================================
def process_image(item):
    """
    处理单个图片文件的完整流程，设计为可被并行调用。
    """
    # 确保子进程能访问全局的H和G矩阵
    global H, G, k_ldpc, LDPC_RATE
    
    try:
        image_base_name, _ = os.path.splitext(item)
        image_path = os.path.join(ROOT_DIR, item)

        # 1. 计算目标比特数
        with Image.open(image_path) as img:
            width, height = img.size
        target_total_bits_n = TARGET_BPP * width * height
        target_bpg_bits_k = target_total_bits_n * LDPC_RATE

        # 2. 寻找最佳QP
        optimal_qp = find_optimal_qp(image_path, target_bpg_bits_k, BPGENC_EXECUTABLE)

        # 3. 最终BPG编码
        bpg_output_path = os.path.join(SAVE_DIR_BPG, image_base_name + '.bin')
        final_bpg_command = [
            BPGENC_EXECUTABLE, '-m', '9', '-b', '8', '-q', str(optimal_qp),
            image_path, '-o', bpg_output_path
        ]
        execute_command(final_bpg_command)
        
        # 4. 读取比特流
        with open(bpg_output_path, 'rb') as f:
            data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
        
        # 5. LDPC编码
        n_blocks = len(data) // k_ldpc
        remainder = len(data) % k_ldpc
        if remainder > 0:
            padding_len = k_ldpc - remainder
            last_block = np.pad(data[n_blocks * k_ldpc:], (0, padding_len), mode='constant')
            data_blocks = np.vstack((data[:n_blocks * k_ldpc:].reshape(-1, k_ldpc), last_block))
        else:
            data_blocks = data[:n_blocks * k_ldpc].reshape(-1, k_ldpc)

        Encoded_bit_blocks = np.vstack([binaryproduct(G, data_blocks[i]) for i in range(data_blocks.shape[0])])
        
        # 6. 保存.mat文件
        mat_save_path = os.path.join(SAVE_DIR_LDPC, image_base_name + '.mat')
        data_to_save = {'encoded_bits': Encoded_bit_blocks}
        savemat(mat_save_path, data_to_save)
        
        final_bpp = Encoded_bit_blocks.size / (width * height)
        print(f">>> [PID:{os.getpid()}] Finished {item}. Final BPP: {final_bpp:.2f}")
        return f"Success: {item}"

    except Exception as e:
        return f"Failed: {item} with error: {e}"

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # --- 初始化LDPC码 (在主进程中完成) ---
    H, G = pyldpc.make_ldpc(n_ldpc, d_v, d_c, systematic=True, sparse=True)
    k_ldpc = G.shape[1]
    n_actual_ldpc = G.shape[0]
    LDPC_RATE = k_ldpc / n_actual_ldpc
    print(f"Designed Rate = 1 - ({d_v}/{d_c}) = {1 - d_v/d_c}")
    print(f"Actual Code Rate = {LDPC_RATE:.2f}")

    # --- 创建输出目录 ---
    os.makedirs(SAVE_DIR_BPG, exist_ok=True)
    os.makedirs(SAVE_DIR_LDPC, exist_ok=True)

    matrix_save_path = os.path.join(SAVE_DIR_LDPC, 'ldpc_matrices.npz')
    np.savez(matrix_save_path, H=H, G=G)

    # --- 准备文件列表 ---
    items_to_process = [
        item for item in sorted(os.listdir(ROOT_DIR)) 
        if item.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]
    # print(f"\nFound {len(items_to_process)} images to process.")
    
    # 4. 使用 multiprocessing.Pool 进行并行处理
    num_processes = min(len(items_to_process), max(1, multiprocessing.cpu_count() - 10))
    # print(f"Starting parallel processing with {num_processes} processes...\n")
    
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_image, items_to_process)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("ALL TASKS COMPLETED")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("Processing Summary:")
    for res in results:
        print(f"  - {res}")
    print("="*50)