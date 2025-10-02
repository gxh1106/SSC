import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_folders, labels, save_dir='.'):
    """
    从多个结果文件夹中读取 psnr_results.csv 文件，
    并绘制 PSNR 和 MS-SSIM 随 SNR 变化的对比曲线图。

    参数:
    results_folders (list): 包含 psnr_results.csv 文件的文件夹路径列表。
    labels (list): 与每个结果文件夹对应的图例标签列表。
    save_dir (str): 保存生成图像的目标文件夹路径。默认为当前目录。
    """
    if len(results_folders) != len(labels):
        raise ValueError("结果文件夹的数量必须与图例标签的数量相同。")

    # 确保保存目录存在，如果不存在则创建
    os.makedirs(save_dir, exist_ok=True)

    # --- 绘制 PSNR vs. SNR 曲线图 ---
    plt.figure(figsize=(10, 6))

    for folder, label in zip(results_folders, labels):
        file_path = os.path.join(folder, 'psnr_results.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            plt.plot(data['SNR (dB)'], data[' PSNR (dB)'], marker='o', linestyle='-', label=label)
        else:
            print(f"警告：在文件夹 '{folder}' 中未找到 'psnr_results.csv' 文件。")

    # 添加图表元素
    plt.title('PSNR vs. SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()
    
    # 构建保存路径并保存图表
    psnr_save_path = os.path.join(save_dir, 'psnr_vs_snr_comparison.png')
    plt.savefig(psnr_save_path)
    print(f"PSNR vs. SNR 对比图已保存到: {os.path.abspath(psnr_save_path)}")
    plt.show()


    # --- 绘制 MS-SSIM vs. SNR 曲线图 ---
    plt.figure(figsize=(10, 6))

    for folder, label in zip(results_folders, labels):
        file_path = os.path.join(folder, 'psnr_results.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            plt.plot(data['SNR (dB)'], data[' MS-SSIM'], marker='s', linestyle='--', label=label)
        else:
            print(f"警告：在文件夹 '{folder}' 中未找到 'psnr_results.csv' 文件。")

    # 添加图表元素
    plt.title('MS-SSIM vs. SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MS-SSIM')
    plt.grid(True)
    plt.legend()
    
    # 构建保存路径并保存图表
    ms_ssim_save_path = os.path.join(save_dir, 'ms_ssim_vs_snr_comparison.png')
    plt.savefig(ms_ssim_save_path)
    print(f"MS-SSIM vs. SNR 对比图已保存到: {os.path.abspath(ms_ssim_save_path)}")
    plt.show()


# --- 使用示例 ---
if __name__ == '__main__':
    subfolder = 'CR_1_8_code4_115k'  # 子文件夹名称
    # 1. 设置包含 .csv 文件的结果文件夹路径
    # 请将这里的路径替换为您自己的实际路径
    result_folders_to_plot = [
        f'output/{subfolder}/test_SwinSSC_VQ',
        f'output/{subfolder}/test_SwinSSC_FAS',
        f'output/{subfolder}/test_SwinSSC_EEP',
        f'output/{subfolder}/test_SwinSSC_UEP_L0',
        f'output/{subfolder}/test_SwinSSC_UEP_L1',
        f'output/{subfolder}/test_SwinSSC_UEP_L2',
        f'output/{subfolder}/test_SwinSSC_UEP_L3',
    ]

    # 2. 为每条曲线设置一个图例标签
    curve_labels = [
        'VQ',
        'FAS',
        'EEP',
        'UEP_L0',
        'UEP_L1',
        'UEP_L2',
        'UEP_L3',
    ]

    # 3. 指定你想要保存图像的文件夹
    output_directory = f'output/{subfolder}'

    # 4. 调用函数，传入结果文件夹、标签和保存路径
    plot_results(result_folders_to_plot, curve_labels, output_directory)