import numpy as np
import pyldpc
import time
import os
import sys  # 导入sys模块以检测操作系统
import subprocess # 导入subprocess模块
from scipy.io import savemat

def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2

def encode(tG, v, seed=None):
    
    n, k = tG.shape

    rnd = np.random.RandomState(seed)
    d = binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    #e = rnd.randn(n) * sigma
    e = rnd.random(x.shape) * sigma
    # y = x + e
    y = x
    return y

def execute_command(command_list):
    """
    一个辅助函数，用于安全地执行外部命令，并处理跨平台问题。
    """
    # 检查操作系统平台
    is_windows = sys.platform == "win32"

    # 如果不是Windows (即Linux, macOS等), 添加wine和xvfb-run
    if not is_windows:
        command_list.insert(0, "wine")
        # command_list.insert(0, "xvfb-run")

    # 设置环境变量，只在非Windows平台静默wine的输出
    my_env = os.environ.copy()
    if not is_windows:
        my_env["WINEDEBUG"] = "fixme-all,err-all"

    try:
        # 使用subprocess.run执行命令
        print(f"Executing: {' '.join(command_list)}")
        subprocess.run(
            command_list,
            env=my_env,
            check=True,         # 如果命令失败（返回非零），则抛出异常
            capture_output=True # 捕获输出，避免其污染主程序输出
        )
    except FileNotFoundError:
        print(f"错误: 命令 '{command_list[0]}' 未找到。请确保它已安装并在系统PATH中。")
        raise
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {' '.join(command_list)}")
        print(f"错误信息: {e.stderr.decode()}")
        raise

n = 50
d_v = 3
d_c = 5     # R = k / n = 1 - d_v / d_c
snr = 10

# 将所有路径和可执行文件名放在这里
# 定义目录名，不带任何斜杠 (跨平台方式)
BASE_DIR_NAME = 'BPG-LDPC'    # ''
DATA_DIR_NAME = 'Kodak24'
# 使用 os.path.join 构建所有路径
# '.' 代表当前目录
BASE_DIR = os.path.join('.', BASE_DIR_NAME, DATA_DIR_NAME)            # 生成 './BPG-LDPC/Kodak24'
ROOT_DIR = os.path.join(BASE_DIR, 'original')         # 生成 './BPG-LDPC/Kodak24/original'
ENCODE_DIR = os.path.join(BASE_DIR, 'encode')         # 生成 './BPG-LDPC/Kodak24/encode'
DECODE_DIR = os.path.join(BASE_DIR, 'decode')         # 生成 './BPG-LDPC/Kodak24/decode'
LDPC_ENCODE_DIR = os.path.join(BASE_DIR, 'ldpc_encode_data')
LDPC_DECODE_DIR = os.path.join(BASE_DIR, 'ldpc_decode_data')
# 可执行文件的路径
BPGENC_EXE = os.path.join('.', BASE_DIR_NAME, "bpgenc.exe") # 生成 './BPG-LDPC/bpgenc.exe'
BPGDEC_EXE = os.path.join('.', BASE_DIR_NAME, "bpgdec.exe") # 生成 './BPG-LDPC/bpgdec.exe'

def main():
    # 生成LDPC码
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    print("H shape:", H.shape)
    print("G shape:", G.shape)
    actual_rate = k / G.shape[0]
    print(f"Designed Rate = 1 - ({d_v}/{d_c}) = {1 - d_v/d_c}")
    print(f"Actual Code Rate (R = k/n): {actual_rate:.2f}\n")

    # 创建所有需要的输出目录
    for path in [ENCODE_DIR, DECODE_DIR, LDPC_ENCODE_DIR, LDPC_DECODE_DIR]:
        os.makedirs(path, exist_ok=True)

    # 遍历所有待处理的图片
    for item in sorted(os.listdir(ROOT_DIR)):
        # 使用os.path.splitext来安全地分离文件名和扩展名
        base_name, _ = os.path.splitext(item)
        full_image_path = os.path.join(ROOT_DIR, item)

        print(f"--- Processing {item} ---")

        # 1. 使用bpgenc进行编码
        bpg_encoded_path = os.path.join(ENCODE_DIR, base_name + '.bin')
        bpgenc_command = [
            BPGENC_EXE, '-m', '9', '-b', '8', '-q', '30',
            full_image_path, '-o', bpg_encoded_path
        ]
        execute_command(bpgenc_command)

        # 2. 读取二进制数据并进行LDPC编码
        with open(bpg_encoded_path, 'rb') as f:
            data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
        print(f"Binary length after BPG: {data.shape}")

        encode_start_time = time.time()

        n_blocks = len(data) // k
        remainder = len(data) % k
        padding_len = 0
        if remainder > 0:
            padding_len = k - remainder
            last_block = np.pad(data[n_blocks * k:], (0, padding_len), mode='constant')
            data_blocks = np.vstack((data[:n_blocks * k].reshape(-1, k), last_block))
        else:
            data_blocks = data[:n_blocks * k].reshape(-1, k)
        
        print(f"Data blocks shape: {data_blocks.shape}")

        Encoded_data_blocks = np.vstack([encode(G, data_blocks[i]) for i in range(data_blocks.shape[0])])
        Encoded_bit_blocks = np.vstack([binaryproduct(G, data_blocks[i]) for i in range(data_blocks.shape[0])])
        
        encoding_time = time.time() - encode_start_time
        print(f"LDPC encoding time: {encoding_time:.4f} seconds")

        ldpc_encoded_path = os.path.join(LDPC_ENCODE_DIR, base_name + '.mat')
        savemat(ldpc_encoded_path, {'encoded_bit': Encoded_bit_blocks})

        # 3. 进行LDPC解码
        decode_start_time = time.time()
        
        y = np.vstack([pyldpc.decode(H, Encoded_data_blocks[i], snr, maxiter=50) for i in range(Encoded_data_blocks.shape[0])])
        decoded_data_bits = np.concatenate([pyldpc.get_message(G, y[i]) for i in range(data_blocks.shape[0])])

        decoding_time = time.time() - decode_start_time
        print(f"LDPC decoding time: {decoding_time:.4f} seconds")

        if padding_len > 0:
            decoded_data_bits = decoded_data_bits[:-padding_len]

        packed_decoded_data = np.packbits(decoded_data_bits)
        
        ldpc_decoded_path = os.path.join(LDPC_DECODE_DIR, base_name + '.bin')
        with open(ldpc_decoded_path, 'wb') as f:
            packed_decoded_data.tofile(f)

        # 4. 使用bpgdec进行解码
        final_image_path = os.path.join(DECODE_DIR, base_name + '.png')
        bpgdec_command = [
            BPGDEC_EXE, '-o', final_image_path, ldpc_decoded_path
        ]
        execute_command(bpgdec_command)
        print(f"--- Finished processing {item} ---\n")

if __name__ == "__main__":
    main()