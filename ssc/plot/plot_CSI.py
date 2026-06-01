import os
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 确保使用 Times New Roman 满足 IEEE 期刊出版要求 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# plt.rcParams['text.usetex'] = True  # 彻底关掉这行，避免报错
plt.rcParams['mathtext.fontset'] = 'stix'  # 开启 stix 字体集，渲染出来的 \varepsilon 和 LaTeX 完全一模一样！

def plot_csi_error_results(root_folder, error_values, save_dir='.', file_suffix=''):
    """
    参数:
    root_folder (str): 基础根目录路径，例如 'output/CSI/bpp2_32C_16E_4D'
    error_values (list): 误差方差数值列表，例如 [0.0, 0.1, 0.2, 0.3, 0.4]。
    save_dir (str): 保存生成图像的目标文件夹路径。
    file_suffix (str): 保存文件名后缀（如 'bpp2'）。
    """
    os.makedirs(save_dir, exist_ok=True)

    axis_label_size = 18  # xy轴标签字体大小
    tick_label_size = 14  # 坐标轴刻度数字大小
    legend_size = 20      # 曲线变多后，图例字体适当调小以防遮挡数据
    line_width = 2        # 线条粗细
    marker_size = 8       # 标记点大小

    plt.rcParams.update({'lines.linewidth': line_width, 'lines.markersize': marker_size})

    markers = ['o', '^', 'D', 's', 'v', 'X', 'p', '*', '<', '>']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    
    def get_zorder(err):
        return 15 - int(err * 10) if err > 0 else 15

    # --- 核心修改：定义 3 种需要对比的误差配置场景 ---
    # 每种场景包含: (场景名称, 文件夹名称拼接逻辑, 对应的图例渲染文本)
    scenarios = [
        ('tr_err', lambda e: f"CSIerr_TX{e:.1f}_RX{e:.1f}", lambda e: r"$\epsilon_t=\epsilon_r=" + f"{e:.1f}$"),
        ('tx_err', lambda e: f"CSIerr_TX{e:.1f}_RX0.0", lambda e: r"$\epsilon_t=" + f"{e:.1f}, \epsilon_r=0.0$"),
        ('rx_err', lambda e: f"CSIerr_TX0.0_RX{e:.1f}", lambda e: r"$\epsilon_t=0.0, \epsilon_r=" + f"{e:.1f}$")
    ]

    # 遍历每种场景开始画图，每种场景各画 PSNR 和 MS-SSIM，总共生成 6 个文件
    for sce_name, folder_func, label_func in scenarios:
        
        # ==========================================
        # 1. 绘制当前场景下的 PSNR vs. SNR 曲线图
        # ==========================================
        plt.figure(figsize=(6, 6))

        for i, err in enumerate(error_values):
            folder_name = folder_func(err)
            
            if err == 0.0:
                label = 'Perfect CSI'
                linestyle = '-'      # 完美信道用坚实实线
            else:
                label = label_func(err) # 动态生成带有 \varepsilon 的标签
                linestyle = '--'     # 误差情况用虚线

            file_path = os.path.join(root_folder, folder_name, 'psnr_results.csv')
            
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                current_marker = markers[i % len(markers)]
                current_color = colors[i % len(colors)]
                current_zorder = get_zorder(err)

                plt.plot(data['SNR (dB)'], data[' PSNR (dB)'], 
                         marker=current_marker, markersize=marker_size, 
                         color=current_color, linestyle=linestyle, linewidth=line_width, 
                         label=label, zorder=current_zorder)
            else:
                print(f"警告: [{sce_name}] 未找到文件 '{file_path}'")

        plt.xlabel('SNR (dB)', fontsize=axis_label_size)
        plt.xlim(0, 20)
        plt.ylabel('PSNR (dB)', fontsize=axis_label_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=legend_size, loc='lower right')
        plt.ylim(bottom=19) 

        # 动态保存当前的 PSNR 图片
        psnr_save_name = f'psnr_{sce_name}_{file_suffix}.pdf' if file_suffix else f'psnr_{sce_name}.pdf'
        psnr_save_path = os.path.join(save_dir, psnr_save_name)
        plt.savefig(psnr_save_path, format='pdf', bbox_inches='tight')
        print(f"已保存: {psnr_save_path}")
        plt.close() # 释放内存


        # ==========================================
        # 2. 绘制当前场景下的 MS-SSIM vs. SNR 曲线图
        # ==========================================
        plt.figure(figsize=(6, 6))

        for i, err in enumerate(error_values):
            folder_name = folder_func(err)
            
            if err == 0.0:
                label = 'Perfect CSI'
                linestyle = '-'
            else:
                label = label_func(err)
                linestyle = '--'

            file_path = os.path.join(root_folder, folder_name, 'psnr_results.csv')
            
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                current_marker = markers[i % len(markers)]
                current_color = colors[i % len(colors)]
                current_zorder = get_zorder(err)

                plt.plot(data['SNR (dB)'], data[' MS-SSIM'], 
                         marker=current_marker, markersize=marker_size, 
                         color=current_color, linestyle=linestyle, linewidth=line_width, 
                         label=label, zorder=current_zorder)
            else:
                print(f"警告: [{sce_name}] 未找到文件 '{file_path}'")

        plt.xlabel('SNR (dB)', fontsize=axis_label_size)
        plt.xlim(0, 20)
        plt.ylabel('MS-SSIM', fontsize=axis_label_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_label_size)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=legend_size, loc='lower right')
        plt.ylim(bottom=0.5) 

        # 动态保存当前的 MS-SSIM 图片
        ms_ssim_save_name = f'ms_ssim_{sce_name}_{file_suffix}.pdf' if file_suffix else f'ms_ssim_{sce_name}.pdf'
        ms_ssim_save_path = os.path.join(save_dir, ms_ssim_save_name)
        plt.savefig(ms_ssim_save_path, format='pdf', bbox_inches='tight')
        print(f"已保存: {ms_ssim_save_path}")
        plt.close()

# --- 使用示例 ---
if __name__ == '__main__':
    subfolder = 'bpp2_32C_16E_4D'  
    file_suffix = subfolder.split('_')[0]  # 提取出 'bpp2' 作为后缀
    
    csi_root_folder = f'output/CSI/{subfolder}'
    error_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    output_directory = f'output/CSI/{subfolder}'

    # 执行绘图
    plot_csi_error_results(csi_root_folder, error_values, output_directory, file_suffix='')