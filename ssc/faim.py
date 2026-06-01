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

    def __init__(self, K: int, N: int, Nr: int, M: int, num_H: int, W: float, L_paths: int, device: str = 'cuda', codebook_size: int = 16, CSI_error_TX: float = 0.0, CSI_error_RX: float = 0.0):
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
            codebook_size (int): 输入索引的码本大小，决定了每个输入索引的比特数。
            CSI_error_TX (float): 发射端CSI误差的标准差。
            CSI_error_RX (float): 接收端CSI误差的标准差。
        """
        super().__init__()

        # --- 1. 参数验证 ---
        assert K <= N and math.log2(M).is_integer() and math.log2(K).is_integer()

        self.K = K
        self.N = N
        self.Nr = Nr
        self.M = M
        self.device = device
        # 假设输入的embedding_indices范围是0-15
        self.bits_per_input_index = int(math.log2(codebook_size))
        self.CSI_error_TX = CSI_error_TX
        self.CSI_error_RX = CSI_error_RX

        # 索引比特数
        self.m_index = int(math.log2(self.K))
        self.num_indexed_combinations = 2**self.m_index

        # 符号比特数
        self.m_sym = int(math.log2(M))
        
        # 每帧传输的总比特数
        self.bits_per_frame = self.m_index + self.m_sym

        # --- 3. 预计算QAM星座和所有可能的发射信号 ---
        H_pool = self._generate_mmwave_channel(Nr, N, W, L_paths, num_H, device) # Shape: (num_H, Nr, N)
        self.Hs_pool = self._select_optimal_channel(H_pool) # Shape: (num_H, Nr, K)
        self.constellation = self._generate_qam_constellation().to(device)
        # 预计算所有可能的发射信号，每个信号都是一个列向量
        self.all_possible_x = self._precompute_all_tx_signals().to(device) # Shape: (K*M, K)

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
            if self.CSI_error_TX > 0.0:
                h_realization_err = math.sqrt(1 - self.CSI_error_TX ** 2) * h_realization + self.CSI_error_TX * math.sqrt(0.5) * (torch.randn_like(h_realization) + 1j * torch.randn_like(h_realization))
            else:
                h_realization_err = h_realization

            max_metric = -float('inf')
            best_indices = None
            # 遍历所有可能的K端口组合
            for i in range(all_combinations.shape[0]):
                current_indices = all_combinations[i, :]
                H_temp = h_realization_err[:, current_indices]
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

    def _bits_to_decimal(self, bits: torch.Tensor) -> torch.Tensor:
        """将一批二进制张量转换为十进制数。"""
        mask = 2**torch.arange(bits.shape[-1] - 1, -1, -1, device=self.device)
        return torch.sum(mask * bits, dim=-1)

    def _decimal_to_bits(self, decimal: torch.Tensor, num_bits: int) -> torch.Tensor:
        """将十进制数转换为指定位数的二进制张量。"""
        mask = 2**torch.arange(num_bits - 1, -1, -1, device=self.device)
        return decimal.unsqueeze(-1).bitwise_and(mask).ne(0).long()

    def forward(self, embedding_indices: torch.Tensor, snr_db: float, idx_H: int = 0, ssc: bool = True, ssc_idx: int = 0, ssc_adapt: bool = False) -> torch.Tensor:
        original_shape = embedding_indices.shape
        BL, depth = original_shape
        
        # --- 1. 编码: 根据输入结构分别构建索引比特流和符号比特流 ---
        if ssc:
            if ssc_adapt:
                # --- [方案 C: ssc=True, ssc_adapt=True, 自适应比例分割] ---
                # 目标是沿着 BL 方向（按列）展开，所以需要先转置，再确保内存连续，最后展平
                # (BL, depth) -> (depth, BL) -> contiguous memory -> (depth * BL)
                all_indices_flat = embedding_indices.transpose(0, 1).contiguous().view(-1)
                total_indices = all_indices_flat.shape[0]
                # 根据 m_index 和 m_sym 的比例来决定分割点
                # 这个比例决定了总信息中有多少应该通过更可靠的端口索引传输
                split_point = int(total_indices * self.m_index / (self.m_index + self.m_sym))
                # 分割成两路十进制索引流
                index_source_decimal = all_indices_flat[:split_point]
                symbol_source_decimal = all_indices_flat[split_point:]
            else:
                # --- [方案 A: ssc=True, ssc_adapt=False, 按特定层分割] ---
                # 动态选择作为端口索引的层
                index_source_decimal = embedding_indices[:, ssc_idx]
                symbol_layer_indices = [i for i in range(depth) if i != ssc_idx]
                symbol_source_decimal = embedding_indices[:, symbol_layer_indices]

                # index_source_decimal = embedding_indices[:, 0]       # Shape: (L,)
                # symbol_source_decimal = embedding_indices[:, 1:]     # Shape: (L, depth-1)
                # index_source_decimal = embedding_indices[:, depth-1]       # Shape: (L,)
                # symbol_source_decimal = embedding_indices[:, 0:depth-1]     # Shape: (L, depth-1)
        
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
            remainder_sym = symbol_bit_stream.shape[0] % self.m_sym
            if remainder_sym != 0:
                padding_sym = torch.zeros(self.m_sym - remainder_sym, dtype=torch.uint8, device=self.device)
                symbol_bit_stream = torch.cat([symbol_bit_stream, padding_sym])
            symbol_bits = symbol_bit_stream.view(-1, self.m_sym)

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
        else:
            # --- [方案 B: ssc=False, 均等保护 (EEP)] ---
            # 将整个 (BL, depth) 索引张量转换为一个连续的比特流
            bit_tensor = self._decimal_to_bits(embedding_indices, self.bits_per_input_index)
            bit_stream = bit_tensor.view(-1)
            original_total_bits = bit_stream.shape[0]

            # 对统一的比特流进行补零
            remainder = original_total_bits % self.bits_per_frame
            if remainder != 0:
                padding = torch.zeros(self.bits_per_frame - remainder, dtype=torch.uint8, device=self.device)
                padded_bit_stream = torch.cat([bit_stream, padding])
            else:
                padded_bit_stream = bit_stream
                
            # 分割成帧
            num_frames = padded_bit_stream.shape[0] // self.bits_per_frame

            # framed_bits = padded_bit_stream.view(num_frames, self.bits_per_frame)
            # index_bits = framed_bits[:, :self.m_index]
            # symbol_bits = framed_bits[:, self.m_index:]

            total_index_bits = num_frames * self.m_index
            # 从 padded_bit_stream 的最前面分割出所有 index bits
            index_bits_flat = padded_bit_stream[:total_index_bits]
            index_bits = index_bits_flat.view(num_frames, self.m_index)
            symbol_bits_flat = padded_bit_stream[total_index_bits:]
            symbol_bits = symbol_bits_flat.view(num_frames, self.bits_per_frame - self.m_index)

            port_indices = self._bits_to_decimal(index_bits)
            symbol_indices = self._bits_to_decimal(symbol_bits)

        # --- 3. 传输: 构建信号 x (K, num_frames), 并通过信道 y = Hx + n ---
        s = self.constellation[symbol_indices]
        # x 的形状是 (K, num_frames)，每一列是一个发射信号向量
        x = torch.zeros(self.K, num_frames, dtype=torch.cfloat, device=self.device)
        # 使用 scatter_ 在 dim=0 (天线维度) 上放置符号
        x.scatter_(0, port_indices.unsqueeze(0), s.unsqueeze(0))
        
        # 预计算解码查找表 y = Hx
        # (Nr, K) @ (K, K*M) -> (Nr, K*M) 
        Hs = self.Hs_pool[idx_H]  # Shape: (Nr, K)
        if self.CSI_error_RX > 0.0:
            Hs_err = math.sqrt(1 - self.CSI_error_RX ** 2) * Hs + self.CSI_error_RX * math.sqrt(0.5) * (torch.randn_like(Hs) + 1j * torch.randn_like(Hs))
        else:
            Hs_err = Hs
        decode_lookup_table = Hs_err @ self.all_possible_x.T # Shape: (Nr, K*M)

        # y_noiseless 的形状是 (Nr, num_frames)
        y_noiseless = Hs @ x

        # signal_power = torch.mean(torch.abs(y_noiseless)**2)
        noise_variance = torch.tensor(1 / (10**(snr_db / 10.0)))
        noise = torch.sqrt(noise_variance / 2) * (torch.randn_like(y_noiseless) + 1j * torch.randn_like(y_noiseless))
        y_noisy = y_noiseless + noise

        # --- 4. 解码: 最大似然检测 ---
        # y_noisy shape: (Nr, num_frames), lookup_table shape: (Nr, K*M)
        # 我们需要比较 y_noisy 的每一列 和 lookup_table 的每一列
        # distances = torch.sum(torch.abs(y_noisy.unsqueeze(1) - decode_lookup_table.unsqueeze(2))**2, dim=0)    # Shape: (K*M, num_frames)
        # decoded_flat_indices = torch.argmin(distances, dim=0)

        batch_size = 2048 
        decoded_flat_indices_list = []
        # 将 y_noisy (Nr, num_frames) 分块处理
        for i in range(0, num_frames, batch_size):
            # 获取当前块
            y_batch = y_noisy[:, i : i + batch_size] # Shape: (Nr, batch_size)
            # 使用广播计算当前块的距离，这将创建一个较小的中间张量
            # (Nr, batch_size, 1) vs (Nr, 1, K*M) -> (Nr, batch_size, K*M) -> (batch_size, K*M)
            distances_batch = torch.sum(torch.abs(y_batch.unsqueeze(2) - decode_lookup_table.unsqueeze(1))**2, dim=0)
            # distances_batch shape: (batch_size, K*M)
            # 找到当前块的最小距离索引
            decoded_indices_batch = torch.argmin(distances_batch, dim=1)
            decoded_flat_indices_list.append(decoded_indices_batch)
        # 将所有块的结果拼接成一个完整的张量
        decoded_flat_indices = torch.cat(decoded_flat_indices_list, dim=0)


        # --- 5. 重建: 将检测结果转换回 (L, depth) 索引 ---
        decoded_port_indices = torch.div(decoded_flat_indices, self.M, rounding_mode='floor')
        decoded_symbol_indices = torch.remainder(decoded_flat_indices, self.M)
        
        if ssc:
            if ssc_adapt:
                # --- [方案 C: ssc=True, ssc_adapt=True, 自适应比例分割重建] ---
                # 1. 从帧索引恢复比特流 (包含 padding)
                decoded_index_bits = self._decimal_to_bits(decoded_port_indices, self.m_index).view(-1)
                decoded_symbol_bits = self._decimal_to_bits(decoded_symbol_indices, self.m_sym).view(-1)
                
                original_idx_indices_len = split_point
                original_sym_indices_len = total_indices - split_point

                original_idx_bits_len = original_idx_indices_len * self.bits_per_input_index
                original_sym_bits_len = original_sym_indices_len * self.bits_per_input_index

                clean_index_bits = decoded_index_bits[:original_idx_bits_len]
                clean_symbol_bits = decoded_symbol_bits[:original_sym_bits_len]

                # 3. 将干净的比特流恢复为十进制索引
                decoded_index_source_decimal = self._bits_to_decimal(clean_index_bits.view(-1, self.bits_per_input_index))
                decoded_symbol_source_decimal = self._bits_to_decimal(clean_symbol_bits.view(-1, self.bits_per_input_index))

                # 4. 重建在编码器中被分割的那个展平的十进制索引流
                reconstructed_flat_indices = torch.cat([decoded_index_source_decimal, decoded_symbol_source_decimal])
                
                # 5. 关键的逆向操作：恢复原始形状
                #    编码器操作: transpose(0, 1).contiguous().view(-1)  (从 (BL, depth) 到列优先的 1D)
                #    解码器逆操作: view(depth, BL).transpose(0, 1)        (从列优先的 1D 回到 (BL, depth))
                output_indices = reconstructed_flat_indices.view(depth, BL).transpose(0, 1).contiguous()

            else:
                # --- [方案 A: ssc=True, ssc_adapt=False, 按特定层分割重建] ---
                decoded_index_bits = self._decimal_to_bits(decoded_port_indices, self.m_index).view(-1)
                decoded_symbol_bits = self._decimal_to_bits(decoded_symbol_indices, self.m_sym).view(-1)
                
                # 截断，移除补零位
                # 假设 ssc_idx 层用于索引流，其余 depth-1 层用于符号流
                symbol_layer_indices = [i for i in range(depth) if i != ssc_idx]
                original_idx_bits_len = BL * self.bits_per_input_index
                original_sym_bits_len = BL * (depth - 1) * self.bits_per_input_index
                
                clean_index_bits = decoded_index_bits[:original_idx_bits_len]
                clean_symbol_bits = decoded_symbol_bits[:original_sym_bits_len]
                
                # 将比特流恢复为十进制索引
                output_indices = torch.zeros(original_shape, dtype=torch.long, device=self.device)
                output_indices[:, ssc_idx] = self._bits_to_decimal(clean_index_bits.view(BL, self.bits_per_input_index))
                output_indices[:, symbol_layer_indices] = self._bits_to_decimal(clean_symbol_bits.view(BL, depth - 1, self.bits_per_input_index))
        else:
            # --- [方案 B: ssc=False, 均等保护 (EEP) 重建] ---
            decoded_index_bits = self._decimal_to_bits(decoded_port_indices, self.m_index)
            decoded_symbol_bits = self._decimal_to_bits(decoded_symbol_indices, self.m_sym)
            # decoded_padded_stream = torch.cat([decoded_index_bits, decoded_symbol_bits], dim=1).view(-1)

            flat_index_bits = decoded_index_bits.view(-1)
            flat_symbol_bits = decoded_symbol_bits.view(-1)
            decoded_padded_stream = torch.cat([flat_index_bits, flat_symbol_bits], dim=0)
            
            # 截断，移除补零位
            decoded_bit_stream = decoded_padded_stream[:original_total_bits]
            
            # 恢复为原始形状
            bit_groups = decoded_bit_stream.view(BL, depth, self.bits_per_input_index)
            output_indices = self._bits_to_decimal(bit_groups)
        return output_indices
    


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
        支持 Square QAM (16, 64, 256...) 和 Rectangular QAM (32, 128...)。
        """
        # 1. 检查 M 是否为 2 的幂
        m_log2 = math.log2(self.M)
        assert m_log2.is_integer(), "M 必须是 2 的幂，例如 4, 16, 32, 64, 128。"
        m_bits = int(m_log2)

        # 2. 确定实部和虚部的比特数 (支持非正方形星座)
        # 如果是奇数比特(如32QAM -> 5 bits)，实部多分1 bit (8x4)
        # n_i: 实部比特数, n_q: 虚部比特数
        n_i = (m_bits + 1) // 2  
        n_q = m_bits - n_i       
        
        len_i = 2 ** n_i  # 实部轴上的点数 (例如 32QAM 为 8)
        len_q = 2 ** n_q  # 虚部轴上的点数 (例如 32QAM 为 4)

        # 3. 定义生成一维格雷码索引的辅助函数
        def get_gray_map(bit_count):
            length = 2 ** bit_count
            gray_map = torch.zeros(length, dtype=torch.long)
            for i in range(length):
                gray_map[i] = i ^ (i >> 1)
            return gray_map

        gray_map_i = get_gray_map(n_i)
        gray_map_q = get_gray_map(n_q)

        # 4. 创建坐标轴点
        # 注意：PAM 坐标点通常为 -(L-1), ..., (L-1)，步长为 2
        axis_points_i = torch.arange(-(len_i - 1), len_i, 2, device=self.device, dtype=torch.float)
        axis_points_q = torch.arange(-(len_q - 1), len_q, 2, device=self.device, dtype=torch.float)

        # 5. 生成二维格雷映射的星座点
        constellation = torch.zeros(self.M, dtype=torch.cfloat, device=self.device)
        
        # 使用嵌套循环填充星座点 (也可以使用广播机制加速，但为了保持原代码逻辑清晰，这里沿用循环)
        # 这里的索引逻辑假设高位比特对应实部，低位比特对应虚部
        for i in range(len_i):      # 实部索引
            for j in range(len_q):  # 虚部索引
                # 获取对应的格雷码索引
                gray_idx_i = gray_map_i[i]
                gray_idx_j = gray_map_q[j]
                
                # 计算最终的十进制索引
                # 索引组合方式：实部索引 * 虚部长度 + 虚部索引
                dec_index = i * len_q + j
                
                # 组合复数星座点
                constellation[dec_index] = axis_points_i[gray_idx_i] + 1j * axis_points_q[gray_idx_j]

        # 6. 归一化以实现单位平均功率
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