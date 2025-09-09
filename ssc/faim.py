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
            K (int): 活动天线/端口的数量。
            N (int): 可用的总天线/端口数量。
            Nr (int): 接收天线的数量。
            M (int): QAM调制的阶数 (例如, 4, 16, 64)。
            H (torch.Tensor): 预计算的信道系数张量。
                               形状应为 (num_realizations, Nr, N)。
            device (str): 计算设备 ('cuda' 或 'cpu')。
        """
        super().__init__()

        # --- 1. 参数验证 ---
        assert K <= N, "活动天线数K不能大于总天线数N"
        assert H.shape[1] == Nr and H.shape[2] == N, f"信道H的形状应为 (any, {Nr}, {N})"
        assert math.log2(M).is_integer(), "调制阶数M必须是2的幂"

        self.K = K
        self.N = N
        self.Nr = Nr
        self.M = M
        self.H = H.to(device).to(torch.cfloat)  # 将信道矩阵移至设备并设为复数
        self.num_channel_realizations = H.shape[0]
        self.device = device

        # --- 2. 预计算索引和符号比特信息 ---
        # 计算所有可能的天线组合 C(N, K)
        self.antenna_combinations = torch.combinations(torch.arange(N), K).to(device)
        self.num_combinations = self.antenna_combinations.shape[0]

        # 索引比特数
        self.m_index = math.floor(math.log2(self.num_combinations))
        self.num_indexed_combinations = 2**self.m_index

        # 符号比特数
        self.m_mod = int(math.log2(M))
        self.num_symbol_bits = self.K * self.m_mod
        
        # 每帧传输的总比特数
        self.bits_per_frame = self.m_index + self.num_symbol_bits

        print("--- FA-IM Channel Initialized ---")
        print(f"  - 天线配置: 选择 {K} 个, 共 {N} 个可用")
        print(f"  - 天线组合数: {self.num_combinations} -> 可索引组合: {self.num_indexed_combinations}")
        print(f"  - 调制: {M}-QAM")
        print(f"  - 每帧比特数: {self.bits_per_frame} (索引: {self.m_index}, 符号: {self.num_symbol_bits})")
        print("---------------------------------")

        # --- 3. 预计算QAM星座和所有可能的发射信号 ---
        self.constellation = self._generate_qam_constellation().to(device)
        self.all_possible_x = self._precompute_all_tx_signals().to(device)
        self.total_patterns = self.all_possible_x.shape[0]

    def _generate_qam_constellation(self) -> torch.Tensor:
        """生成一个单位平均功率的QAM星座点集。"""
        sqrt_m = int(math.sqrt(self.M))
        qam_points = torch.zeros(self.M, dtype=torch.cfloat)
        idx = 0
        for i in range(sqrt_m):
            for j in range(sqrt_m):
                qam_points[idx] = complex(2*i - sqrt_m + 1, 2*j - sqrt_m + 1)
                idx += 1
        # 归一化以实现单位平均功率
        return qam_points / torch.sqrt(torch.mean(torch.abs(qam_points)**2))

    def _precompute_all_tx_signals(self) -> torch.Tensor:
        """预计算所有可能的发射信号向量x的查找表，用于ML检测。"""
        num_symbol_patterns = self.M**self.K
        total_patterns = self.num_indexed_combinations * num_symbol_patterns
        
        all_x = torch.zeros(total_patterns, self.N, dtype=torch.cfloat)

        # 生成所有可能的K个符号的组合
        symbol_indices = torch.from_numpy(
            np.array(list(product(range(self.M), repeat=self.K)))
        )
        symbol_patterns = self.constellation[symbol_indices] # (M**K, K)

        # 遍历所有被索引的天线组合
        for i in range(self.num_indexed_combinations):
            start_idx = i * num_symbol_patterns
            end_idx = (i + 1) * num_symbol_patterns
            
            # 获取当前的天线索引
            active_antennas = self.antenna_combinations[i]
            
            # 将所有符号模式放置在活动天线上
            all_x[start_idx:end_idx, active_antennas] = symbol_patterns
            
        return all_x

    def _bits_to_decimal(self, bits: torch.Tensor) -> torch.Tensor:
        """将一批二进制张量转换为十进制数。"""
        mask = 2**torch.arange(bits.shape[-1] - 1, -1, -1, device=self.device)
        return torch.sum(mask * bits, dim=-1)

    def _decimal_to_bits(self, decimal: torch.Tensor, num_bits: int) -> torch.Tensor:
        """将十进制数转换为指定位数的二进制张量。"""
        mask = 2**torch.arange(num_bits - 1, -1, -1, device=self.device)
        return decimal.unsqueeze(-1).bitwise_and(mask).ne(0).long()

    def forward(self, binary_tensor: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        对输入的比特流进行FA-IM传输仿真。

        Args:
            binary_tensor (torch.Tensor): 输入的二进制张量，形状为 [batch_size, num_bits]。
                                          num_bits 必须是 self.bits_per_frame 的整数倍。
            snr_db (float): 当前信道的信噪比 (dB)。

        Returns:
            torch.Tensor: 经过信道和解码后的带噪比特流，形状与输入相同。
        """
        # --- 1. 输入整形和比特分割 ---
        assert binary_tensor.dim() == 2, "输入张量应为2维 [batch_size, num_bits]"
        batch_size, num_bits = binary_tensor.shape
        assert num_bits % self.bits_per_frame == 0, "总比特数必须是每帧比特数的整数倍"
        
        num_frames = num_bits // self.bits_per_frame
        
        # [B, N_bits] -> [B * N_frames, bits_per_frame]
        frames = binary_tensor.reshape(-1, self.bits_per_frame)
        total_frames = frames.shape[0]
        
        # 分割为索引比特和符号比特
        index_bits = frames[:, :self.m_index]
        symbol_bits = frames[:, self.m_index:]

        # --- 2. 比特到索引和符号的映射 ---
        # 索引比特 -> 天线组合索引
        decimal_indices = self._bits_to_decimal(index_bits)
        
        # 符号比特 -> QAM符号
        symbol_bits_reshaped = symbol_bits.reshape(total_frames, self.K, self.m_mod)
        symbol_indices = self._bits_to_decimal(symbol_bits_reshaped)
        qam_symbols = self.constellation[symbol_indices] # [total_frames, K]

        # --- 3. 构建发射信号向量 x ---
        x = torch.zeros(total_frames, self.N, dtype=torch.cfloat, device=self.device)
        active_antennas = self.antenna_combinations[decimal_indices] # [total_frames, K]
        
        # 使用 scatter_ 将符号放置在活动天线位置上
        # 这是比for循环更高效的PyTorch方法
        x.scatter_(1, active_antennas, qam_symbols)
        
        # --- 4. 信道传输 ---
        # 为每帧随机选择一个信道实现
        chan_indices = torch.randint(0, self.num_channel_realizations, (total_frames,))
        H_eff = self.H[chan_indices] # [total_frames, Nr, N]

        # 计算噪声
        snr_linear = 10**(snr_db / 10.0)
        noise_variance = 1 / snr_linear
        
        noise = torch.sqrt(torch.tensor(noise_variance / 2)) * \
                (torch.randn_like(H_eff[:, :, 0], dtype=torch.float) + 1j * torch.randn_like(H_eff[:, :, 0], dtype=torch.float))

        # 接收信号 y = Hx + n
        # [total_frames, Nr, N] @ [total_frames, N, 1] -> [total_frames, Nr, 1]
        y = torch.bmm(H_eff, x.unsqueeze(-1)).squeeze(-1) + noise # [total_frames, Nr]

        # --- 5. 最大似然 (ML) 检测 ---
        # [total_frames, Nr, N] @ [N, total_patterns] -> [total_frames, Nr, total_patterns]
        all_possible_y_clean = torch.matmul(H_eff, self.all_possible_x.T)
        
        # 计算欧氏距离的平方
        # y: [total_frames, Nr, 1]
        # all_possible_y_clean: [total_frames, Nr, total_patterns]
        # 距离: [total_frames, total_patterns]
        distances = torch.sum(torch.abs(y.unsqueeze(-1) - all_possible_y_clean)**2, dim=1)
        
        # 找到最小距离对应的索引
        detected_indices = torch.argmin(distances, dim=1)

        # --- 6. 解码为比特 ---
        num_symbol_patterns = self.M**self.K
        
        # 解码天线索引和符号索引
        decoded_antenna_idx = torch.div(detected_indices, num_symbol_patterns, rounding_mode='floor')
        decoded_symbol_pattern_idx = torch.remainder(detected_indices, num_symbol_patterns)

        # 索引 -> 比特
        decoded_index_bits = self._decimal_to_bits(decoded_antenna_idx, self.m_index)

        # 符号 -> 比特
        # 需要从一维的符号模式索引恢复为K个独立的符号索引
        symbol_indices_base_m = torch.zeros(total_frames, self.K, dtype=torch.long, device=self.device)
        temp_idx = decoded_symbol_pattern_idx
        for i in range(self.K - 1, -1, -1):
            symbol_indices_base_m[:, i] = torch.remainder(temp_idx, self.M)
            temp_idx = torch.div(temp_idx, self.M, rounding_mode='floor')
            
        decoded_symbol_bits = self._decimal_to_bits(symbol_indices_base_m, self.m_mod)
        decoded_symbol_bits = decoded_symbol_bits.reshape(total_frames, -1)
        
        # --- 7. 重组输出 ---
        decoded_frames = torch.cat([decoded_index_bits, decoded_symbol_bits], dim=1)
        
        # [B * N_frames, bits_per_frame] -> [B, N_bits]
        output_binary_tensor = decoded_frames.reshape(batch_size, -1)
        
        return output_binary_tensor


# --- 示例用法 ---
if __name__ == '__main__':
    # 定义系统参数
    K_sys, N_sys, Nr_sys, M_sys = 2, 4, 4, 4  # (选2个,共4个), 4个接收天线, 4-QAM
    device_sys = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 生成随机信道矩阵 H
    num_realizations = 100
    H_sys = torch.randn(num_realizations, Nr_sys, N_sys, dtype=torch.cfloat) / np.sqrt(N_sys)

    # 实例化信道模块
    fa_im_channel = FA_IM_Channel(K=K_sys, N=N_sys, Nr=Nr_sys, M=M_sys, H=H_sys, device=device_sys)

    # 生成一批随机比特流 (batch_size=10, 10帧)
    batch_size = 10
    num_frames = 10
    total_bits_to_send = batch_size * num_frames * fa_im_channel.bits_per_frame
    input_bits = torch.randint(0, 2, (batch_size, num_frames * fa_im_channel.bits_per_frame)).to(device_sys)
    
    # 设置信噪比
    snr_db = 15.0

    # 通过信道传输
    output_bits = fa_im_channel(input_bits, snr_db)

    # 计算误比特率 (BER)
    num_errors = torch.sum(input_bits != output_bits).item()
    ber = num_errors / total_bits_to_send
    
    print(f"\n--- Simulation Result ---")
    print(f"SNR: {snr_db} dB")
    print(f"Total bits sent: {total_bits_to_send}")
    print(f"Bit errors: {num_errors}")
    print(f"Bit Error Rate (BER): {ber:.2e}")