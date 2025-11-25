# flake8: noqa
import torch
def calculate_p_from_snr_db(snr_db):
    """
    根据给定的SNR（单位：dB），计算BPSK在AWGN信道下的比特错误概率p。

    参数:
        snr_db (torch.Tensor or float): 一个或多个信噪比的值，以dB为单位。

    返回:
        torch.Tensor: 对应的比特错误概率 p。
    """
    # 确保输入是torch.Tensor
    if not isinstance(snr_db, torch.Tensor):
        snr_db = torch.tensor(snr_db, dtype=torch.float32)

    # 1. 将SNR从dB转换为线性值 (SNR_linear = 10^(SNR_dB / 10))
    snr_linear = 10.0 ** (snr_db / 10.0)

    # 2. 对于BPSK调制，Eb/N0 在数值上等于 SNR_linear
    eb_n0 = snr_linear

    # 3. 使用公式 p = 0.5 * erfc(sqrt(Eb/N0)) 计算比特错误概率
    # torch.erfc 是PyTorch中的互补误差函数
    p = 0.5 * torch.erfc(torch.sqrt(eb_n0))
    
    return p

snr = torch.tensor([-2,1,4,7,10,13,16])
p = calculate_p_from_snr_db(snr)
print(p)
# print(torch.round(p, decimals=3)) # 保留3位精度