import torch
import torch.nn as nn
import numpy as np
import math
from itertools import product
import torch.nn.functional as F

class FA_IM_Channel(nn.Module):
    """
    PyTorch模块，用于模拟流体天线索引调制（FA-IM）信道。

    该模块将输入的二进制张量通过FA-IM方案进行加扰、信道传输和最大似然解码，
    最终输出带噪声的二进制张量。
    """

    def __init__(self, K: int, N: int, Nr: int, M: int, num_H: int, W: float, L_paths: int, device: str = 'cpu'):
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
        assert K <= N and math.log2(M).is_integer() and math.log2(K).is_integer()

        self.K = K
        self.N = N
        self.Nr = Nr
        self.M = M
        self.device = device

        # 索引比特数
        self.m_index = int(math.log2(self.K))
        self.num_indexed_combinations = 2**self.m_index

        # 符号比特数
        self.m_sym = int(math.log2(M))
        
        # 每帧传输的总比特数
        self.bits_per_frame = self.m_index + self.m_sym
        self.num_total_signals = 2**self.bits_per_frame

        # --- 3. 预计算QAM星座和所有可能的发射信号 ---
        H_pool = self._generate_mmwave_channel(Nr, N, W, L_paths, num_H, device) # Shape: (num_H, Nr, N)
        self.Hs_pool = self._select_optimal_channel(H_pool) # Shape: (num_H, Nr, K)
        self.constellation = self._generate_qam_constellation().to(device)
        # 预计算所有可能的发射信号，每个信号都是一个列向量
        self.all_possible_x = self._precompute_all_tx_signals().to(device) # Shape: (K*M, K)
        self.all_possible_bits = self._precompute_all_possible_bits().to(device) # Shape: (K*M, bits_per_frame)

    def _generate_mmwave_channel(self, Nr, N, W, L, num_realizations = 1, device = 'cuda') -> torch.Tensor:
        """
        生成一维流体天线（发射端）到标准ULA（接收端）的毫米波信道矩阵。

        该函数基于几何多径信道模型。

        Args:
            Nr (int): 接收端天线数量 (标准ULA)。
            N (int): 发射端端口数量 (流体天线ULA)。
            W (float): 发射端流体天线的总宽度 (单位：米)。
            L (int): 多径信道的路径数量。
            num_realizations (int, optional): 生成的信道实现数量（批处理大小）。默认为 1。

        Returns:
            torch.Tensor: 生成的信道矩阵 H，形状为 (num_realizations, Nr, N)。
        """
        # 接收端 (Rx): 标准半波长间隔的ULA
        dr = 1 / 2
        rx_pos = torch.arange(Nr, device=device, dtype=torch.float32) * dr
        # 发射端 (Tx): 在宽度W内均匀分布的流体天线ULA
        dt = W / (N - 1) if N > 1 else 0
        tx_pos = torch.arange(N, device=device, dtype=torch.float32) * dt
        # 角度范围 [-pi/2, pi/2] (阵列前方)
        AoD = (torch.rand(num_realizations, L, device=device) * math.pi) - (math.pi / 2)
        AoA = (torch.rand(num_realizations, L, device=device) * math.pi) - (math.pi / 2)
        gains = (torch.randn(num_realizations, L, device=device, dtype=torch.cfloat)) / math.sqrt(L)
        # --- 计算阵列响应 (Steering Vectors) ---
        # sin(AoA) shape: (num_realizations, L) -> (num_realizations, L, 1)
        # rx_pos shape: (Nr,) -> (1, 1, Nr)
        # 广播后计算相位，结果 shape: (num_realizations, L, Nr)
        rx_phases = 2 * math.pi * torch.sin(AoA).unsqueeze(2) * rx_pos.view(1, 1, -1)
        a_r = torch.exp(1j * rx_phases)
        # sin(AoD) shape: (num_realizations, L) -> (num_realizations, L, 1)
        # tx_pos shape: (N,) -> (1, 1, N)
        # 结果 shape: (num_realizations, L, N)
        tx_phases = 2 * math.pi * torch.sin(AoD).unsqueeze(2) * tx_pos.view(1, 1, -1)
        a_t = torch.exp(1j * tx_phases)
        # --- 构建信道矩阵 ---
        # H = sum over L of [ gain_l * a_r_l * a_t_l^H ]
        # a_r -> (num_realizations, L, Nr, 1)
        # a_t_H -> (num_realizations, L, 1, N)
        # path_response shape: (num_realizations, L, Nr, N)
        path_response = a_r.unsqueeze(3) @ a_t.unsqueeze(2).conj()
        # gains -> (num_realizations, L, 1, 1)
        # 加权并沿路径维度求和
        H = torch.sum(gains.view(num_realizations, L, 1, 1) * path_response, dim=1)
        return H

    def _select_optimal_channel(self, H_pool) -> torch.Tensor:
        """从H_pool中选择最优的K个端口以最大化信道容量。"""
        all_combinations = torch.combinations(torch.arange(self.N), self.K)
        optimal_channels = []
        # 对池中的每一个信道实现进行操作
        for h_realization in H_pool:
            max_metric = -float('inf')
            best_indices = None
            # 遍历所有可能的K端口组合
            for i in range(all_combinations.shape[0]):
                current_indices = all_combinations[i, :]
                H_temp = h_realization[:, current_indices]
                metric_matrix = torch.eye(self.K, device=self.device) + (1/self.K) * (H_temp.T.conj() @ H_temp)
                _sign, logabsdet = torch.slogdet(metric_matrix)
                metric = logabsdet.item()

                if metric > max_metric:
                    max_metric = metric
                    best_indices = current_indices
            
            # 存储这个信道实现的最优子信道
            optimal_channels.append(h_realization[:, best_indices])
        
        # 将列表堆叠成一个张量
        return torch.stack(optimal_channels, dim=0)
    
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

    def _precompute_all_tx_signals(self) -> torch.Tensor:
        """预计算所有可能的发射信号向量x的查找表 (K*M, K)。"""
        total_patterns = self.K * self.M
        all_x = torch.zeros(total_patterns, self.K, dtype=torch.cfloat)
        for port_idx in range(self.K):
            start_idx = port_idx * self.M
            end_idx = (port_idx + 1) * self.M
            all_x[start_idx:end_idx, port_idx] = self.constellation
        return all_x

    def _precompute_all_possible_bits(self) -> torch.Tensor:
        """
        预计算所有可能传输比特模式的查找表。
        第i行是整数i的二进制表示。
        """
        # 创建从0到num_total_signals-1的整数
        decimal_indices = torch.arange(self.num_total_signals, device=self.device, dtype=torch.long)
        
        # 将每个整数转换为其比特模式
        all_bits = self._decimal_to_bits(decimal_indices, self.bits_per_frame)
        
        return all_bits

    def _bits_to_decimal(self, bits: torch.Tensor) -> torch.Tensor:
        """将一批二进制张量转换为十进制数。"""
        mask = 2**torch.arange(bits.shape[-1] - 1, -1, -1, device=self.device)
        return torch.sum(mask * bits, dim=-1)

    def _decimal_to_bits(self, decimal: torch.Tensor, num_bits: int) -> torch.Tensor:
        """将十进制数转换为指定位数的二进制张量。"""
        mask = 2**torch.arange(num_bits - 1, -1, -1, device=self.device)
        return decimal.unsqueeze(-1).bitwise_and(mask).ne(0).long()

    def forward(self, bits: torch.Tensor, snr_db: float, idx_H: int = 0, batch_size: int = 4096) -> torch.Tensor:
        """
        Processes bits through the FA-IM channel and returns LLRs.
        Uses batching to manage memory for large bitstreams.
        """
        bits = bits.to(self.device)
        original_len = bits.shape[0]

        remainder = original_len % self.bits_per_frame
        if remainder != 0:
            padding_len = self.bits_per_frame - remainder
            # 创建与输入张量相同类型和设备的补零张量
            padding = torch.zeros(padding_len, dtype=bits.dtype, device=self.device)
            # 将补零拼接到原始比特流末尾
            padded_bits = torch.cat([bits, padding], dim=0)
        else:
            # 如果长度刚好，则无需补零
            padded_bits = bits
        
        num_total_bits = padded_bits.shape[0]
        num_frames = num_total_bits // self.bits_per_frame
        bit_frames = padded_bits.reshape(num_frames, self.bits_per_frame)

        all_llrs = []
        # # select channel from the pool for the batch
        # H = self.Hs_pool[idx_H].to(self.device) # Shape: (Nr, K)
        # # all_y_hat = H @ self.all_possible_x.T -> (Nr, K*M)
        # all_y_hat = H @ self.all_possible_x.T

        for i in range(0, num_frames, batch_size):
            batch_frames = bit_frames[i:i+batch_size]
            current_batch_size = batch_frames.shape[0]

            # 1. Map bits to transmitted signal x
            x_indices = self._bits_to_decimal(batch_frames)
            x = self.all_possible_x[x_indices]  # Shape: (batch_size, K)

            # 2. Simulate channel: y = Hx + n
            snr_linear = 10 ** (snr_db / 10.0)
            noise_variance = 1 / snr_linear
            
            # Generate complex noise
            noise = torch.sqrt(torch.tensor(noise_variance / 2.0)) * \
                    (torch.randn(current_batch_size, self.Nr, device=self.device) + 1j * torch.randn(current_batch_size, self.Nr, device=self.device))
            
            # Randomly select channel realizations for the batch
            h_indices = torch.randint(0, self.Hs_pool.shape[0], (current_batch_size,), device=self.device)
            H = self.Hs_pool[h_indices]  # 形状: [batch_size, Nr, K]

            all_y_hat = H @ self.all_possible_x.T # 形状: [batch_size, Nr, num_total_signals]

            # Calculate received signal y
            y = (H @ x.unsqueeze(-1)).squeeze(-1) + noise # Shape: (batch_size, Nr)

            # 3. Maximum Likelihood (ML) Detection and LLR Calculation
            # Calculate likelihoods for all possible transmitted signals
            # Euclidean distance: ||y - Hx||^2
            # y -> (batch_size, Nr, 1)
            # all_y_hat -> (batch_size, Nr, K*M)
            # dist_sq -> (batch_size, K*M)
            dist_sq = torch.sum(torch.abs(y.unsqueeze(2) - all_y_hat)**2, dim=1)
            
            # Likelihoods are proportional to exp(-dist^2 / noise_var)
            likelihoods = F.softmax(-dist_sq / noise_variance, dim=1) # Use softmax for numerical stability

            # 4. Calculate LLRs for each bit
            # batch_llrs = torch.zeros_like(batch_frames, dtype=torch.float64)
            # for j in range(self.bits_per_frame):
            #     # Sum of likelihoods where j-th bit is 0
            #     mask_0 = (self.all_possible_bits[:, j] == 0)
            #     sum_p0 = torch.sum(likelihoods * mask_0, dim=1)
            #     # Sum of likelihoods where j-th bit is 1
            #     mask_1 = (self.all_possible_bits[:, j] == 1)
            #     sum_p1 = torch.sum(likelihoods * mask_1, dim=1)
            #     # LLR = log(P0/P1)
            #     batch_llrs[:, j] = torch.log(sum_p0 / (sum_p1 + 1e-20) + 1e-20) # Add epsilon for stability

            bits_float = self.all_possible_bits.float()
            sum_p1 = likelihoods @ bits_float
            sum_p0 = likelihoods @ (1 - bits_float) # 总概率为1，所以P0 = 1 - P1不完全成立，因为是不同比特位的概率。这里应该用总和减去P1
            # 更稳健的写法是:
            # total_likelihood_sum = torch.sum(likelihoods, dim=1, keepdim=True)
            # sum_p0 = total_likelihood_sum - sum_p1
            # 但直接用(1-bits_float)在数学上是等价且更高效的。
            batch_llrs = torch.log(sum_p0.clamp(min=1e-20) / sum_p1.clamp(min=1e-20))

            all_llrs.append(batch_llrs)

        final_llrs_padded = torch.cat(all_llrs, dim=0).flatten()

        return final_llrs_padded[:original_len]
    
    
    

class FA_SISO_Channel(nn.Module):
    """
    PyTorch模块，用于模拟流体天线（FA-SISO）信道。

    该模块将输入的二进制张量通过FA-SISO方案进行加扰、信道传输和最大似然解码，
    最终输出带噪声的二进制张量。
    """
    def __init__(self, N: int, Nr: int, M: int, num_H: int, W: float, L_paths: int, device: str = 'cuda', codebook_size: int = 16):
        super().__init__()

        # --- 1. 参数验证 ---
        assert math.log2(M).is_integer()

        self.N = N
        self.Nr = Nr
        self.M = M
        self.device = device
        # 假设输入的embedding_indices范围是0-15
        self.bits_per_input_index = int(math.log2(codebook_size))

        # 符号比特数
        self.m_sym = int(math.log2(M))
        
        # 每帧传输的总比特数
        self.bits_per_frame = self.m_sym

        # --- 3. 预计算QAM星座和所有可能的发射信号 ---
        H_pool = self._generate_mmwave_channel(Nr, N, W, L_paths, num_H, device) # Shape: (num_H, Nr, N)
        self.Hs_pool = self._select_optimal_channel(H_pool) # Shape: (num_H, Nr, 1)
        self.constellation = self._generate_qam_constellation().to(device)
        # 预计算所有可能的无噪声接收信号 y = h * s
        # (num_H, Nr, 1) @ (1, M) -> (num_H, Nr, M)
        self.decode_lookup_table_pool = self.Hs_pool @ self.constellation.unsqueeze(0)

    def _generate_mmwave_channel(self, Nr, N, W, L, num_realizations = 1, device = 'cuda') -> torch.Tensor:
        # 接收端 (Rx): 标准半波长间隔的ULA
        dr = 1 / 2
        rx_pos = torch.arange(Nr, device=device, dtype=torch.float32) * dr
        # 发射端 (Tx): 在宽度W内均匀分布的流体天线ULA
        dt = W / (N - 1) if N > 1 else 0
        tx_pos = torch.arange(N, device=device, dtype=torch.float32) * dt
        # 角度范围 [-pi/2, pi/2] (阵列前方)
        AoD = (torch.rand(num_realizations, L, device=device) * math.pi) - (math.pi / 2)
        AoA = (torch.rand(num_realizations, L, device=device) * math.pi) - (math.pi / 2)
        gains = (torch.randn(num_realizations, L, device=device, dtype=torch.cfloat)) / math.sqrt(L)
        # --- 计算阵列响应 (Steering Vectors) ---
        # sin(AoA) shape: (num_realizations, L) -> (num_realizations, L, 1)
        # rx_pos shape: (Nr,) -> (1, 1, Nr)
        # 广播后计算相位，结果 shape: (num_realizations, L, Nr)
        rx_phases = 2 * math.pi * torch.sin(AoA).unsqueeze(2) * rx_pos.view(1, 1, -1)
        a_r = torch.exp(1j * rx_phases)
        # sin(AoD) shape: (num_realizations, L) -> (num_realizations, L, 1)
        # tx_pos shape: (N,) -> (1, 1, N)
        # 结果 shape: (num_realizations, L, N)
        tx_phases = 2 * math.pi * torch.sin(AoD).unsqueeze(2) * tx_pos.view(1, 1, -1)
        a_t = torch.exp(1j * tx_phases)
        # --- 构建信道矩阵 ---
        # H = sum over L of [ gain_l * a_r_l * a_t_l^H ]
        # a_r -> (num_realizations, L, Nr, 1)
        # a_t_H -> (num_realizations, L, 1, N)
        # path_response shape: (num_realizations, L, Nr, N)
        path_response = a_r.unsqueeze(3) @ a_t.unsqueeze(2).conj()
        # gains -> (num_realizations, L, 1, 1)
        # 加权并沿路径维度求和
        H = torch.sum(gains.view(num_realizations, L, 1, 1) * path_response, dim=1)
        return H

    def _select_optimal_channel(self, H_pool) -> torch.Tensor:
        """
        从H_pool中为每个信道实现选择L2范数最大的单个端口。
        """
        # H_pool shape: (num_H, Nr, N)
        # 计算每个端口（列）的L2范数
        # norms shape: (num_H, N)
        norms = torch.linalg.vector_norm(H_pool, dim=1)
        
        # 找到每个信道实现中范数最大的端口的索引
        # best_port_indices shape: (num_H,)
        best_port_indices = torch.argmax(norms, dim=1)
        
        # 使用 gather 从 H_pool 中高效地选出对应的列
        # 索引需要被扩展以匹配 gather 的要求
        # (num_H,) -> (num_H, 1, 1) -> (num_H, Nr, 1)
        indices_to_gather = best_port_indices.view(-1, 1, 1).expand(-1, self.Nr, 1)
        
        # H_best shape: (num_H, Nr, 1)
        H_best = H_pool.gather(2, indices_to_gather)
        return H_best
    
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

    def _bits_to_decimal(self, bits: torch.Tensor) -> torch.Tensor:
        """将一批二进制张量转换为十进制数。"""
        mask = 2**torch.arange(bits.shape[-1] - 1, -1, -1, device=self.device)
        return torch.sum(mask * bits, dim=-1)

    def _decimal_to_bits(self, decimal: torch.Tensor, num_bits: int) -> torch.Tensor:
        """将十进制数转换为指定位数的二进制张量。"""
        mask = 2**torch.arange(num_bits - 1, -1, -1, device=self.device)
        return decimal.unsqueeze(-1).bitwise_and(mask).ne(0).long()

    def forward(self, embedding_indices: torch.Tensor, snr_db: float, idx_H: int = 0, **kwargs) -> torch.Tensor:
        original_shape = embedding_indices.shape
        BL, depth = original_shape
        
        # --- 1. 选择信道和对应的解码表 ---
        H_selected = self.Hs_pool[idx_H]              # Shape: (Nr, 1)
        decode_lookup_table = self.decode_lookup_table_pool[idx_H] # Shape: (Nr, M)

        # --- 2. 统一编码、补零与分帧 ---
        bit_tensor = self._decimal_to_bits(embedding_indices, self.bits_per_input_index)
        bit_stream = bit_tensor.view(-1)
        original_total_bits = bit_stream.shape[0]

        remainder = original_total_bits % self.bits_per_frame
        if remainder != 0:
            padding = torch.zeros(self.bits_per_frame - remainder, dtype=torch.uint8, device=self.device)
            padded_bit_stream = torch.cat([bit_stream, padding])
        else:
            padded_bit_stream = bit_stream
            
        num_frames = padded_bit_stream.shape[0] // self.bits_per_frame
        if num_frames == 0: return torch.zeros_like(embedding_indices)
        
        symbol_bits = padded_bit_stream.view(num_frames, self.m_sym)
        symbol_indices = self._bits_to_decimal(symbol_bits)

        # --- 3. 传输 y = h*s + n ---
        s = self.constellation[symbol_indices] # Shape: (num_frames,)
        
        # (Nr, 1) @ (1, num_frames) -> (Nr, num_frames)
        y_noiseless = H_selected @ s.unsqueeze(0)

        noise_variance = torch.tensor(1 / (10**(snr_db / 10.0)))
        noise = torch.sqrt(noise_variance / 2) * (torch.randn_like(y_noiseless) + 1j * torch.randn_like(y_noiseless))
        y_noisy = y_noiseless + noise

        # --- 4. 解码 (分块处理以节省内存) ---
        batch_size = 2048
        decoded_symbol_indices_list = []
        for i in range(0, num_frames, batch_size):
            y_batch = y_noisy[:, i : i + batch_size]
            # (Nr, batch_size, 1) vs (Nr, 1, M) -> (Nr, batch_size, M) -> (batch_size, M)
            distances_batch = torch.sum(torch.abs(y_batch.unsqueeze(2) - decode_lookup_table.unsqueeze(1))**2, dim=0)
            decoded_indices_batch = torch.argmin(distances_batch, dim=1)
            decoded_symbol_indices_list.append(decoded_indices_batch)
        
        decoded_symbol_indices = torch.cat(decoded_symbol_indices_list, dim=0)

        # --- 5. 重建 ---
        decoded_padded_stream = self._decimal_to_bits(decoded_symbol_indices, self.m_sym).view(-1)
        decoded_bit_stream = decoded_padded_stream[:original_total_bits]
        
        bit_groups = decoded_bit_stream.view(BL, depth, self.bits_per_input_index)
        output_indices = self._bits_to_decimal(bit_groups)
        
        return output_indices
    

# --- 示例用法 ---
if __name__ == '__main__':
    # 定义系统参数
    BL = 2       # 批处理大小
    depth = 4      # 量化深度

    K = 4         # 活动端口数
    M = 64         # 星座大小
    N = 16         # 总可用端口数
    Nr = 8        # 接收天线数
    snr_db = 25.0  # 信噪比

    # 定义信道物理参数
    num_H = 10     # 创建10个不同的信道实现
    W = 2  # 流体天线宽度为2个波长
    L_paths = 20      # 20条多径

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 关键更改在这里: 初始化方式改变 ---
    fa_bit_system = FA_IM_Channel(
        K=K, M=M, N=N, Nr=Nr, 
        num_H=num_H, W=W, L_paths=L_paths, device=device
    )

    # 创建一批随机的输入索引
    original_indices = torch.randint(0, 16, (BL, depth), device=device)

    # --- 关键更改在这里: forward调用方式改变 ---
    # 假设我们想用第3个信道 (索引从0开始) 进行传输
    channel_to_use_idx = 3
    recovered_indices = fa_bit_system(original_indices, snr_db, channel_to_use_idx)

    # 评估性能
    errors = (original_indices != recovered_indices).sum()
    error_rate = errors / (BL * depth)

    print(f"\n系统参数: K={K}, M={M}, N={N}, Nr={Nr}")
    print(f"使用的信道索引: {channel_to_use_idx}")
    print("-" * 20)
    print(f"原始索引 (前5行):\n{original_indices[:5]}")
    print(f"恢复索引 (前5行):\n{recovered_indices[:5]}")
    print(f"总索引错误率: {error_rate.item():.6f}")