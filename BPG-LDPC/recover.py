import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- 配置参数 (只需要与你主脚本中结果相关的部分) ---
# 这些参数需要和你卡住的那个程序完全一致！
TARGET_BPP = 3.0
bpp_suffix = f"_bpp{TARGET_BPP}"

n_ldpc = 1944
d_v = 23
d_c = 24

BASE_DIR = './BPG-LDPC' 
DATA_DIR = 'Kodak24'
RESULTS_DIR = os.path.join(BASE_DIR, DATA_DIR, f'results_resize_{n_ldpc}_{d_v}_{d_c}', f'results{bpp_suffix}')

SNR_START = 0
SNR_INTERVAL = 2
SNR_END = 24 + SNR_INTERVAL
SNR_range = np.arange(SNR_START, SNR_END, SNR_INTERVAL)
# ---------------------------------------------------------

def main():
    """
    从 live_results.csv 恢复数据，并完成汇总和绘图。
    """
    live_results_path = os.path.join(RESULTS_DIR, 'live_results.csv')
    
    print(f"正在从以下文件恢复数据: {live_results_path}")

    if not os.path.exists(live_results_path):
        print(f"错误: 结果文件 '{live_results_path}' 不存在。请检查路径是否正确。")
        sys.exit(1)

    # --- 这部分代码与你主脚本的最后部分几乎完全一样 ---
    # 1. 使用 pandas 读取已保存的实时结果文件
    try:
        results_df = pd.read_csv(live_results_path)
        print(f"成功读取 {len(results_df)} 条记录。")
    except pd.errors.EmptyDataError:
        print("错误：结果文件为空，无法进行汇总。")
        sys.exit(1)
        
    # 如果文件最后一行可能不完整，可以加上这句来处理
    # results_df.dropna(how='all', inplace=True)

    # 2. 数据清洗和处理
    # 删除psnr列为空的行，这些是解码失败或未完成的任务
    original_count = len(results_df)
    results_df.dropna(subset=['psnr'], inplace=True)
    cleaned_count = len(results_df)
    print(f"数据清洗：移除了 {original_count - cleaned_count} 条不完整的或失败的记录。")
    
    # 确保数据类型正确
    results_df['snr'] = results_df['snr'].astype(int)
    results_df['psnr'] = results_df['psnr'].astype(float)
    results_df['ms_ssim'] = results_df['ms_ssim'].astype(float)

    # 3. 按SNR分组并计算平均值
    print("正在按SNR分组并计算平均性能...")
    summary = results_df.groupby('snr')[['psnr', 'ms_ssim']].mean().reset_index()
    summary.rename(columns={'psnr': 'Avg PSNR (dB)', 'ms_ssim': 'Avg MS-SSIM'}, inplace=True)
    
    avg_psnr_list = summary['Avg PSNR (dB)'].tolist()
    avg_ms_ssim_list = summary['Avg MS-SSIM'].tolist()

    # 4. 打印汇总结果到控制台
    print("\n--- Average Performance on Kodak24 Dataset (Recovered) ---")
    for index, row in summary.iterrows():
        print(f"SNR: {int(row['snr']):2d} dB | Avg PSNR: {row['Avg PSNR (dB)']:.4f} dB | Avg MS-SSIM: {row['Avg MS-SSIM']:.4f}")
    
    # 5. 保存汇总结果到新的CSV文件
    summary_file_path = os.path.join(RESULTS_DIR, 'results.csv')
    summary.to_csv(summary_file_path, index=False)
    print(f"\n汇总结果已保存到: {summary_file_path}")
    
    # 6. 绘制并保存曲线图
    # 绘制PSNR
    plt.figure(figsize=(10, 7))
    plt.plot(summary['snr'], avg_psnr_list, marker='o', linestyle='-', label=f'BPG+LDPC (FA-IM, BPP={TARGET_BPP})')
    plt.title('PSNR vs. SNR Performance (Recovered)')
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Average Peak Signal-to-Noise Ratio (PSNR) [dB]')
    plt.xticks(SNR_range)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plot_path_psnr = os.path.join(RESULTS_DIR, 'PSNR_vs_SNR_curve.png')
    plt.savefig(plot_path_psnr)
    print(f"PSNR 曲线图已保存到: {plot_path_psnr}")

    # 绘制MS-SSIM
    plt.figure(figsize=(10, 7))
    plt.plot(summary['snr'], avg_ms_ssim_list, marker='s', linestyle='--', color='crimson', label=f'BPG+LDPC (FA-IM, BPP={TARGET_BPP})')
    plt.title('MS-SSIM vs. SNR Performance (Recovered)')
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Average MS-SSIM')
    plt.xticks(SNR_range)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plot_path_msssim = os.path.join(RESULTS_DIR, 'MS_SSIM_vs_SNR_curve.png')
    plt.savefig(plot_path_msssim)
    print(f"MS-SSIM 曲线图已保存到: {plot_path_msssim}")


if __name__ == "__main__":
    main()