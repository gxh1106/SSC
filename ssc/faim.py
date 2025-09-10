import torch
import torch.nn as nn
import numpy as np
import math
from itertools import product

class FA_IM_Channel(nn.Module):
    """
    PyTorch模块，用于模拟流体天线索引调制（FA-IM）信道。

    该模块将输入的二进制张量通过FA-IM方案进行加扰、信道传输和最大似然解码，
    最终输出带噪声的二进制张量。
    """

    def __init__(self, K: int, N: int, Nr: int, M: int, H: torch.Tensor, device: str = 'cuda'):
        """
        初始化FA-IM信道模块。

        Args:
            K (int): 活动端口的数量。
            N (int): 可用的总端口数量。
            Nr (int): 接收天线的数量。
            M (int): QAM调制的阶数 (例如, 4, 16, 64)。
            H (torch.Tensor): 预计算的信道系数张量。
                               形状应为 (num_realizations, Nr, N)。
            device (str): 计算设备 ('cuda' 或 'cpu')。
        """
        super().__init__()

        # --- 1. 参数验证 ---
        assert K <= N, "活动端口数K不能大于总端口数N"
        assert H.shape[1] == Nr and H.shape[2] == N, f"信道H的形状应为 (any, {Nr}, {N})"
        assert math.log2(M).is_integer(), "调制阶数M必须是2的幂"
        assert math.log2(K).is_integer(), "活动端口数K必须是2的幂"

        self.K = K
        self.N = N
        self.Nr = Nr
        self.M = M
        self.H_pool = H.to(device).to(torch.cfloat)
        self.num_channel_realizations = H.shape[0]
        self.device = device
        # 假设输入的embedding_indices范围是0-15
        self.bits_per_input_index = 4

        # 索引比特数
        self.m_index = int(math.log2(self.K))
        self.num_indexed_combinations = 2**self.m_index

        # 符号比特数
        self.m_mod = int(math.log2(M))
        self.num_symbol_bits = self.m_mod
        
        # 每帧传输的总比特数
        self.bits_per_frame = self.m_index + self.num_symbol_bits

        # --- 3. 预计算QAM星座和所有可能的发射信号 ---
        self.H_optimal = self._select_optimal_channel() # Shape: (Nr, K)
        self.constellation = self._generate_qam_constellation().to(device)
        # 预计算所有可能的发射信号，每个信号都是一个列向量
        self.all_possible_x = self._precompute_all_tx_signals().to(device) # Shape: (K*M, K)
        # 预计算解码查找表 y = Hx
        # (Nr, K) @ (K, K*M) -> (Nr, K*M) 
        self.decode_lookup_table = self.H_optimal @ self.all_possible_x.T

    def _generate_qam_constellation(self) -> torch.Tensor:
        """
        生成一个单位平均功率且采用格雷映射的QAM星座点集。
        """
        assert math.sqrt(self.M).is_integer(), "M必须是完全平方数，例如4, 16, 64。"
        
        # 1. 生成一维格雷码索引
        m_1d = int(math.log2(math.sqrt(self.M)))
        gray_map_1d = torch.zeros(2**m_1d, dtype=torch.long)
        for i in range(2**m_1d):
            gray_map_1d[i] = i ^ (i >> 1)

        # 2. 创建星座点网格
        sqrt_m = int(math.sqrt(self.M))
        axis_points = torch.arange(-(sqrt_m - 1), sqrt_m, 2, device=self.device)
        
        # 3. 生成二维格雷映射的星座点
        constellation = torch.zeros(self.M, dtype=torch.cfloat, device=self.device)
        for i in range(sqrt_m): # 实部索引
            for j in range(sqrt_m): # 虚部索引
                # 将常规二进制索引 (i, j) 映射到格雷码索引
                gray_i = gray_map_1d[i]
                gray_j = gray_map_1d[j]
                
                # 计算最终的十进制索引
                dec_index = i * sqrt_m + j
                
                constellation[dec_index] = axis_points[gray_i] + 1j * axis_points[gray_j]
                
        # 4. 归一化以实现单位平均功率
        power = torch.mean(torch.abs(constellation)**2)
        return constellation / torch.sqrt(power)
    
    # def _generate_qam_constellation(self) -> torch.Tensor:
    #     """生成一个单位平均功率的QAM星座点集。"""
    #     sqrt_m = int(math.sqrt(self.M))
    #     qam_points = torch.zeros(self.M, dtype=torch.cfloat)
    #     idx = 0
    #     for i in range(sqrt_m):
    #         for j in range(sqrt_m):
    #             qam_points[idx] = complex(2*i - sqrt_m + 1, 2*j - sqrt_m + 1)
    #             idx += 1
    #     # 归一化以实现单位平均功率
    #     return qam_points / torch.sqrt(torch.mean(torch.abs(qam_points)**2))

    def _precompute_all_tx_signals(self) -> torch.Tensor:
        """预计算所有可能的发射信号向量x的查找表 (K*M, K)。"""
        total_patterns = self.K * self.M
        all_x = torch.zeros(total_patterns, self.K, dtype=torch.cfloat)
        for port_idx in range(self.K):
            start_idx = port_idx * self.M
            end_idx = (port_idx + 1) * self.M
            all_x[start_idx:end_idx, port_idx] = self.constellation
        return all_x

    def _select_optimal_channel(self) -> torch.Tensor:
        """从H_pool中选择最优的K个端口以最大化信道容量。"""
        all_combinations = torch.combinations(torch.arange(self.N), self.K)
        max_metric = -1.0
        best_indices = None
        
        # 为简化，我们只评估第一个信道实现来选择最优端口组合
        H_eval = self.H_pool[0]
        for i in range(all_combinations.shape[0]):
            current_indices = all_combinations[i, :]
            H_temp = H_eval[:, current_indices]
            metric_matrix = torch.eye(self.K, device=self.device) + (1/self.K) * (H_temp.T.conj() @ H_temp)
            metric = torch.slogdet(metric_matrix).logabsdet.item()
            if metric > max_metric:
                max_metric = metric
                best_indices = current_indices
        
        return self.H_pool[0][:, best_indices]
    
    def _bits_to_decimal(self, bits: torch.Tensor) -> torch.Tensor:
        """将一批二进制张量转换为十进制数。"""
        mask = 2**torch.arange(bits.shape[-1] - 1, -1, -1, device=self.device)
        return torch.sum(mask * bits, dim=-1)

    def _decimal_to_bits(self, decimal: torch.Tensor, num_bits: int) -> torch.Tensor:
        """将十进制数转换为指定位数的二进制张量。"""
        mask = 2**torch.arange(num_bits - 1, -1, -1, device=self.device)
        return decimal.unsqueeze(-1).bitwise_and(mask).ne(0).long()

    def forward(self, embedding_indices: torch.Tensor, snr_db: float) -> torch.Tensor:
        original_shape = embedding_indices.shape
        L, depth = original_shape
        
        # --- 1. 编码: 根据输入结构分别构建索引比特流和符号比特流 ---
        # index_source_decimal = embedding_indices[:, 0]       # Shape: (L,)
        # symbol_source_decimal = embedding_indices[:, 1:]     # Shape: (L, depth-1)
        index_source_decimal = embedding_indices[:, depth-1]       # Shape: (L,)
        symbol_source_decimal = embedding_indices[:, 0:depth-1]     # Shape: (L, depth-1)

        # 将十进制索引转换为比特流
        index_bit_stream = self._decimal_to_bits(index_source_decimal, self.bits_per_input_index).view(-1)
        symbol_bit_stream = self._decimal_to_bits(symbol_source_decimal, self.bits_per_input_index).view(-1)

        # --- 2. 分帧与补零 ---
        # 对索引比特流进行补零和分帧
        remainder_idx = index_bit_stream.shape[0] % self.m_index
        if remainder_idx != 0:
            padding_idx = torch.zeros(self.m_index - remainder_idx, dtype=torch.uint8, device=self.device)
            index_bit_stream = torch.cat([index_bit_stream, padding_idx])
        index_bits = index_bit_stream.view(-1, self.m_index)

        # 对符号比特流进行补零和分帧
        remainder_sym = symbol_bit_stream.shape[0] % self.m_mod
        if remainder_sym != 0:
            padding_sym = torch.zeros(self.m_mod - remainder_sym, dtype=torch.uint8, device=self.device)
            symbol_bit_stream = torch.cat([symbol_bit_stream, padding_sym])
        symbol_bits = symbol_bit_stream.view(-1, self.m_mod)

        # 确保帧数匹配（以较长的为准，短的补全）
        num_frames = max(index_bits.shape[0], symbol_bits.shape[0])
        port_indices = self._bits_to_decimal(index_bits)
        symbol_indices = self._bits_to_decimal(symbol_bits)
        if port_indices.shape[0] < num_frames:
            padding = torch.zeros(num_frames - port_indices.shape[0], dtype=torch.long, device=self.device)
            port_indices = torch.cat([port_indices, padding])
        if symbol_indices.shape[0] < num_frames:
            padding = torch.zeros(num_frames - symbol_indices.shape[0], dtype=torch.long, device=self.device)
            symbol_indices = torch.cat([symbol_indices, padding])

        # --- 3. 传输: 构建信号 x (K, num_frames), 并通过信道 y = Hx + n ---
        s = self.constellation[symbol_indices]
        # x 的形状是 (K, num_frames)，每一列是一个发射信号向量
        x = torch.zeros(self.K, num_frames, dtype=torch.cfloat, device=self.device)
        # 使用 scatter_ 在 dim=0 (天线维度) 上放置符号
        x.scatter_(0, port_indices.unsqueeze(0), s.unsqueeze(0))
        
        # y_noiseless 的形状是 (Nr, num_frames)
        y_noiseless = self.H_optimal @ x
        
        signal_power = torch.mean(torch.abs(y_noiseless)**2)
        noise_variance = signal_power / (10**(snr_db / 10.0))
        noise = torch.sqrt(noise_variance / 2) * (torch.randn_like(y_noiseless) + 1j * torch.randn_like(y_noiseless))
        y_noisy = y_noiseless + noise

        # --- 4. 解码: 最大似然检测 ---
        # y_noisy shape: (Nr, num_frames), lookup_table shape: (Nr, K*M)
        # 我们需要比较 y_noisy 的每一列 和 lookup_table 的每一列
        distances = torch.sum(torch.abs(y_noisy.unsqueeze(1) - self.decode_lookup_table.unsqueeze(2))**2, dim=0)    # Shape: (K*M, num_frames)
        decoded_flat_indices = torch.argmin(distances, dim=0)

        # --- 5. 重建: 将检测结果转换回 (L, depth) 索引 ---
        decoded_port_indices = torch.div(decoded_flat_indices, self.M, rounding_mode='floor')
        decoded_symbol_indices = torch.remainder(decoded_flat_indices, self.M)
        
        decoded_index_bits = self._decimal_to_bits(decoded_port_indices, self.m_index).view(-1)
        decoded_symbol_bits = self._decimal_to_bits(decoded_symbol_indices, self.m_mod).view(-1)
        
        # 截断，移除补零位
        original_idx_bits_len = L * self.bits_per_input_index
        original_sym_bits_len = L * (depth - 1) * self.bits_per_input_index
        
        clean_index_bits = decoded_index_bits[:original_idx_bits_len]
        clean_symbol_bits = decoded_symbol_bits[:original_sym_bits_len]
        
        # 将比特流恢复为十进制索引
        output_indices = torch.zeros(original_shape, dtype=torch.long, device=self.device)
        output_indices[:, 0] = self._bits_to_decimal(clean_index_bits.view(L, self.bits_per_input_index))
        if depth > 1:
            output_indices[:, 1:] = self._bits_to_decimal(clean_symbol_bits.view(L, depth - 1, self.bits_per_input_index))

        return output_indices
    

# --- 示例用法 ---
if __name__ == '__main__':
    # 定义系统参数
    L = 2       # 批处理大小
    depth = 4      # 量化深度

    K = 4         # 活动端口数 
    M = 64         # 星座大小 (例如，64-QAM，需要 6 比特符号)
    N = 16         # 总可用端口数
    Nr = 8        # 接收天线数
    snr_db = 20.0  # 信噪比

    # 每帧传输的总比特数 = log2(K) + log2(M) = 2 + 6 = 8 bits
    # 原始数据每帧比特数 = L * depth * 4 = 1024 * 4 * 4 = 16384 bits
    # 将传输的帧数 = 16384 // 8 = 2048 帧
    # 会有 16384 % 8 = 0 比特被截断

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 创建随机的信道矩阵H
    H_pool = torch.randn(1, Nr, N, dtype=torch.cfloat) 

    # 3. 创建FA-IM比特流系统实例
    fa_bit_system = FA_IM_Channel(K=K, M=M, N=N, Nr=Nr, H=H_pool, device=device)

    # 4. 创建一批随机的输入索引 (模拟RQ-VAE的输出)
    original_indices = torch.randint(0, 16, (L, depth), device=device)

    # 5. 通过系统进行传输和解码
    recovered_indices = fa_bit_system(original_indices, snr_db)

    # 6. 评估性能
    # 注意：由于截断，最后一个索引可能不完整，但这里的填充逻辑可以处理
    errors = (original_indices != recovered_indices).sum()
    total_indices = L * depth
    error_rate = errors / total_indices

    print(f"系统参数: K={K} (m_index={fa_bit_system.m_index}), M={M} (m_mod={fa_bit_system.m_mod})")
    print(f"每帧传输 {fa_bit_system.bits_per_frame} 比特")
    print("-" * 20)
    print(f"原始索引 (前5行):\n{original_indices[:5]}")
    print(f"恢复索引 (前5行):\n{recovered_indices[:5]}")
    print(f"总索引错误率: {error_rate.item():.6f}")