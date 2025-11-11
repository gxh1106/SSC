import os
import subprocess
import numpy as np
from pyldpc import make_ldpc, encode, get_message

def bytes_to_bit_array(byte_data: bytes) -> np.ndarray:
    """将字节数据高效转换为Numpy位数组 (0s and 1s)"""
    # 从字节缓冲区创建uint8数组
    int_array = np.frombuffer(byte_data, dtype=np.uint8)
    # 将每个8位整数解包成8个比特
    bit_array = np.unpackbits(int_array)
    return bit_array

def bpg_encode_image(input_path, output_path, qp=28):
    """
    调用外部 bpgenc 命令来对单个图像进行编码。
    
    Args:
        input_path (str): 输入图像文件的路径。
        output_path (str): BPG 输出文件的保存路径。
        qp (int): 量化参数 (Quantization Parameter)。值越小，质量越高，文件越大。
    """
    print(f"--- [BPG] 开始编码: {os.path.basename(input_path)} ---")
    command = [
        'bpgenc',
        '-q', str(qp),      # 设置量化参数
        '-o', output_path,  # 指定输出文件
        input_path          # 指定输入文件
    ]
    
    try:
        # 执行命令
        # check=True 表示如果命令返回非零退出码（即发生错误），则会抛出异常
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[BPG] 编码成功, 文件保存在: {output_path}")
        return True
    except FileNotFoundError:
        print("[错误] 'bpgenc' 命令未找到。请确保 BPG 工具已安装并在您的 PATH 中。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[错误] bpgenc 编码失败。返回码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return False

def ldpc_encode_file(input_path, output_path, k, rate=0.5):
    """
    读取一个文件（如BPG文件），并对其二进制内容进行LDPC编码。
    
    Args:
        input_path (str): 输入文件的路径。
        output_path (str): LDPC编码后数据(.npy格式)的保存路径。
        k (int): LDPC编码的信息块长度 (information block size)。
        rate (float): LDPC 码率 (R = k/n)。
    """
    print(f"--- [LDPC] 开始编码: {os.path.basename(input_path)} ---")
    
    # 1. 读取BPG文件的原始二进制数据
    try:
        with open(input_path, 'rb') as f:
            bpg_binary_data = f.read()
    except IOError as e:
        print(f"[错误] 无法读取文件 {input_path}: {e}")
        return

    # 2. 将二进制数据转换为比特流 (numpy array of 0s and 1s)
    bits_stream = bytes_to_bit_array(bpg_binary_data)
    L = len(bits_stream)
    print(f"[LDPC] BPG文件大小: {len(bpg_binary_data)} 字节 = {L} 比特")

    # 3. 定义LDPC码参数
    n = int(k / rate) # 码字长度 (codeword length)
    d_v, d_c = 2, 4   # 校验矩阵中每列/每行的1的个数 (经验值)

    print(f"[LDPC] 码参数: n={n}, k={k}, R={rate}")

    # 4. 构建LDPC码的校验矩阵H和生成矩阵G
    # H 是稀疏的，G 通常是稠密的
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    # 5. 对数据进行分块和编码
    encoded_blocks = []
    num_blocks = (L + k - 1) // k  # 计算需要多少个块 (向上取整)

    for i in range(num_blocks):
        # 获取当前信息块
        start_idx = i * k
        end_idx = start_idx + k
        message_block = bits_stream[start_idx:end_idx]
        
        # 如果是最后一个块，且长度不足k，则用0进行填充 (Padding)
        current_len = len(message_block)
        if current_len < k:
            padding = np.zeros(k - current_len, dtype=int)
            message_block = np.concatenate((message_block, padding))
            
        # 使用生成矩阵 G 进行编码
        encoded_block = encode(G, message_block, sparse=False)
        encoded_blocks.append(encoded_block)

    # 6. 将所有编码后的块拼接成最终的码字
    final_encoded_data = np.concatenate(encoded_blocks)
    
    # 7. 保存为 .npy 文件，便于后续处理
    np.save(output_path, final_encoded_data)
    print(f"[LDPC] 编码成功, {num_blocks}个块被编码。")
    print(f"[LDPC] 最终数据长度: {len(final_encoded_data)} 比特, 保存在: {output_path}")


def main():
    # --- 用户配置 ---
    KODAK_DIR = './datasets/Kodak24' # Kodak数据集路径
    BPG_OUTPUT_DIR = './output/bpg_output'                     # BPG文件输出目录
    LDPC_OUTPUT_DIR = './output/ldpc_output'                   # LDPC编码文件输出目录

    BPG_QP = 28          # BPG量化参数
    LDPC_BLOCK_SIZE = 1024 # LDPC信息块大小 (k)
    LDPC_RATE = 0.5      # LDPC码率 (R)
    # --- 配置结束 ---

    # 创建输出目录
    os.makedirs(BPG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LDPC_OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(KODAK_DIR):
        print(f"[错误] Kodak数据集目录未找到: {KODAK_DIR}")
        print("请确保已下载数据集并正确配置 KODAK_DIR 变量。")
        return

    # 遍历Kodak数据集中的所有png图片
    for filename in sorted(os.listdir(KODAK_DIR)):
        if filename.lower().endswith('.png'):
            # 定义文件路径
            base_name = os.path.splitext(filename)[0]
            input_image_path = os.path.join(KODAK_DIR, filename)
            bpg_file_path = os.path.join(BPG_OUTPUT_DIR, f"{base_name}.bpg")
            ldpc_file_path = os.path.join(LDPC_OUTPUT_DIR, f"{base_name}.npy")

            print(f"\n======= 处理图片: {filename} =======")
            
            # 第1步：BPG编码
            if bpg_encode_image(input_image_path, bpg_file_path, qp=BPG_QP):
                # 第2步：LDPC编码
                ldpc_encode_file(bpg_file_path, ldpc_file_path, k=LDPC_BLOCK_SIZE, rate=LDPC_RATE)

if __name__ == '__main__':
    main()