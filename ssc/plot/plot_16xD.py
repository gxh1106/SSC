import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_folders, labels, save_dir='.', file_suffix=''):
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

    axis_label_size = 16  # xy轴标签字体大小 ("SNR (dB)")
    tick_label_size = 14  # 坐标轴刻度数字大小 ("10", "20")
    legend_size = 12      # 图例字体大小
    line_width = 2        # 线条粗细
    marker_size = 8       # 标记点大小

    plt.rcParams.update({'lines.linewidth': line_width, 'lines.markersize': marker_size})

    markers = ['o', '^', 'D', 's', 'v', 'X', 'p', '*', '<', '>']

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    
    # 如果你想手动指定简单的颜色，也可以这样写：
    # colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y'] 

    # 数字越大，画在越上面（不被遮挡）。(网格线通常是 1，所以要大于1)
    zorder_config = { 'sDAC with FA-IM': 9, 'SSC': 10, 'BPG+LDPC': 2, 'SSC w/o IM': 6, 'SSC w/o SS': 8 }
    default_zorder = 2

    # --- 绘制 PSNR vs. SNR 曲线图 ---
    plt.figure(figsize=(6, 6))

    for i, (folder, label) in enumerate(zip(results_folders, labels)):
        file_path = os.path.join(folder, 'psnr_results.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)

            current_marker = markers[i % len(markers)]
            current_color = colors[i % len(colors)]
            current_zorder = zorder_config.get(label, default_zorder)
            plt.plot(data['SNR (dB)'], data[' PSNR (dB)'], marker=current_marker, markersize=marker_size, color=current_color, linestyle='-', linewidth=line_width, label=label, zorder=current_zorder)
        else:
            print(f"警告：在文件夹 '{folder}' 中未找到 'psnr_results.csv' 文件。")

    # 添加图表元素
    plt.xlabel('SNR (dB)', fontsize=axis_label_size)
    plt.ylabel('PSNR (dB)', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
    plt.grid(True, linestyle='--', alpha=0.6)
    # loc='best' 让系统自动寻找遮挡最少的位置, 'lower right', 'upper left'
    plt.legend(fontsize=legend_size, loc='lower center')
    plt.ylim(bottom=18) 

    # 构建保存路径并保存图表
    psnr_save_path = os.path.join(save_dir, f'psnr_{file_suffix}.pdf')
    plt.savefig(psnr_save_path, format='pdf', bbox_inches='tight')
    print(f"PSNR vs. SNR 对比图已保存到: {os.path.abspath(psnr_save_path)}")
    plt.show()


    # --- 绘制 MS-SSIM vs. SNR 曲线图 ---
    plt.figure(figsize=(6, 6))

    for i, (folder, label) in enumerate(zip(results_folders, labels)):
        file_path = os.path.join(folder, 'psnr_results.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            current_marker = markers[i % len(markers)]
            current_color = colors[i % len(colors)]
            current_zorder = zorder_config.get(label, default_zorder)
            plt.plot(data['SNR (dB)'], data[' MS-SSIM'], marker=current_marker, markersize=marker_size, color=current_color, linestyle='-', linewidth=line_width, label=label, zorder=current_zorder)
        else:
            print(f"警告：在文件夹 '{folder}' 中未找到 'psnr_results.csv' 文件。")

    # 添加图表元素
    plt.xlabel('SNR (dB)', fontsize=axis_label_size)
    plt.ylabel('MS-SSIM', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=legend_size, loc='lower center')
    plt.ylim(bottom=0.5) 

    # 构建保存路径并保存图表
    ms_ssim_save_path = os.path.join(save_dir, f'ms_ssim_{file_suffix}.pdf')
    plt.savefig(ms_ssim_save_path, format='pdf', bbox_inches='tight')
    print(f"MS-SSIM vs. SNR 对比图已保存到: {os.path.abspath(ms_ssim_save_path)}")
    plt.show()


# --- 使用示例 ---
if __name__ == '__main__':
    subfolder = 'bpp1_32C_16E_2D'  # 子文件夹名称
    file_suffix = subfolder.split('_')[0]  # 提取 bpp 部分作为文件后缀
    # 1. 设置包含 .csv 文件的结果文件夹路径
    # 请将这里的路径替换为您自己的实际路径
    result_folders_to_plot = [
        f'output/16xD/{subfolder}/SSC',
        f'output/16xD/{subfolder}/SSC_woIM',
        f'output/16xD/{subfolder}/SSC_woSS',
        f'output/16xD/{subfolder}/sDAC_FA-IM',
        f'output/16xD/{subfolder}/BPG+LDPC',
    ]

    # 2. 为每条曲线设置一个图例标签
    curve_labels = [
        'SSC',
        'SSC w/o IM',
        'SSC w/o SS',
        'sDAC with FA-IM',
        'BPG+LDPC',
    ]

    # 3. 指定你想要保存图像的文件夹
    output_directory = f'output/16xD'

    # 4. 调用函数，传入结果文件夹、标签和保存路径
    plot_results(result_folders_to_plot, curve_labels, output_directory, file_suffix=file_suffix)