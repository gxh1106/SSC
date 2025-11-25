import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_ke_performance(root_dir, target_snrs, target_codebook, save_dir='.'):
    """
    根据指定的 SNR 和 码本大小 (如 '16E'), 绘制性能随 Ke 变化的曲线。

    参数:
    root_dir (str): 包含结果子文件夹的根目录 (例如 'output/Ke')
    target_snr (float): 想要查询的 SNR 值 (例如 18)
    target_codebook (str): 码本大小前缀 (例如 '16E' 或 '64E')
    save_dir (str): 图片保存路径
    """
    
    # 存储提取到的数据: 列表中的元素为元组 (Ke, PSNR, MS-SSIM)
    extracted_data = []

    # 1. 遍历根目录下的所有文件夹
    if not os.path.exists(root_dir):
        print(f"错误: 路径 '{root_dir}' 不存在。")
        return

    subfolders = os.listdir(root_dir)
    
    for folder_name in subfolders:
        folder_path = os.path.join(root_dir, folder_name)
        
        # 过滤条件：必须是文件夹，且名称以指定的码本大小开头 (如 "16E_")
        if os.path.isdir(folder_path) and folder_name.startswith(target_codebook + '_'):
            
            # --- 解析文件名提取 Ke 值 ---
            # 假设格式为 "16E_Ke16"，我们按 "_" 分割，取第二部分 "Ke16"
            try:
                parts = folder_name.split('_')
                ke_part = [p for p in parts if p.startswith('Ke')][0] # 找到 KeX 部分
                ke_value = int(ke_part.replace('Ke', '')) # 去掉 'Ke' 变成数字
            except (IndexError, ValueError):
                print(f"警告: 无法从文件夹名 '{folder_name}' 中解析 Ke 值，已跳过。")
                continue

            # --- 读取 CSV 文件 ---
            csv_path = os.path.join(folder_path, 'psnr_results.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # 清理列名空格 (防止 " PSNR (dB)" 这种带空格的情况)
                    df.columns = [c.strip() for c in df.columns]
                    
                    # 查找指定 SNR 的行
                    # 注意：这里使用浮点数近似匹配，防止精度问题 (例如 18.0 vs 18)
                    # 遍历用户指定的每一个 SNR
                    for snr in target_snrs:
                        # 模糊匹配 SNR (防止浮点数精度问题)
                        row = df[abs(df['SNR (dB)'] - snr) < 0.001]
                        
                        if not row.empty:
                            extracted_data.append({
                                'Ke': ke_value,
                                'SNR': snr,
                                'PSNR': row['PSNR (dB)'].values[0],
                                'MS-SSIM': row['MS-SSIM'].values[0]
                            })
                except Exception as e:
                    print(f"读取文件 '{csv_path}' 出错: {e}")
            else:
                print(f"警告: '{folder_name}' 下没有 psnr_results.csv")

    # 2. 如果没有提取到数据，直接返回
    if not extracted_data:
        print("未找到任何有效数据，请检查路径或 SNR 设置。")
        return

    # 转为 DataFrame 方便处理
    df_all = pd.DataFrame(extracted_data)
    
    # 获取所有出现过的 Ke 值，用于设置 X 轴刻度
    all_ke_values = sorted(df_all['Ke'].unique())

    # --- 绘图通用设置 ---
    os.makedirs(save_dir, exist_ok=True)
    
    # 字体和样式设置
    axis_label_size = 18
    tick_label_size = 12
    legend_size = 18
    line_width = 2
    marker_size = 9
    
    # 设置全局字体 (可选)
    # plt.rcParams['font.family'] = 'serif' 
    # 定义一组 Marker 和 颜色，确保不同 SNR 曲线区分明显
    markers = ['o', 'v', '^', 's', 'D', 'X', '*']
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

    plt.figure(figsize=(6, 6))
    
    # 对列表中的每个 SNR 进行循环绘图
    for i, snr in enumerate(sorted(target_snrs)):
        # 筛选出当前 SNR 的数据，并按 Ke 排序
        subset = df_all[df_all['SNR'] == snr].sort_values(by='Ke')
        
        if not subset.empty:
            style_idx = i % len(markers) # 循环使用样式
            plt.plot(subset['Ke'], subset['PSNR'], 
                     marker=markers[style_idx], 
                     color=colors[style_idx],
                     linestyle='-', linewidth=line_width, markersize=marker_size,
                     label=f'SNR = {snr} dB')

    plt.xlabel('Codeword Dimension ($d_e$)', fontsize=axis_label_size)
    plt.ylabel('PSNR (dB)', fontsize=axis_label_size)
    plt.tick_params(axis='x', labelsize=tick_label_size)
    plt.tick_params(axis='y', labelsize=tick_label_size+2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=legend_size)
    plt.xticks(all_ke_values) # 强制显示所有 Ke 刻度

    save_path_psnr = os.path.join(save_dir, f'{target_codebook}_PSNR.pdf')
    plt.savefig(save_path_psnr, format='pdf', bbox_inches='tight')
    print(f"\nPSNR 图已保存: {save_path_psnr}")
    plt.show()

    plt.figure(figsize=(6, 6))
    
    for i, snr in enumerate(sorted(target_snrs)):
        subset = df_all[df_all['SNR'] == snr].sort_values(by='Ke')
        
        if not subset.empty:
            style_idx = i % len(markers)
            plt.plot(subset['Ke'], subset['MS-SSIM'], 
                     marker=markers[style_idx], 
                     color=colors[style_idx],
                     linestyle='-', linewidth=line_width, markersize=marker_size,
                     label=f'SNR = {snr} dB')

    plt.xlabel('Codeword Dimension ($d_e$)', fontsize=axis_label_size)
    plt.ylabel('MS-SSIM', fontsize=axis_label_size)
    # plt.tick_params(axis='both', labelsize=tick_label_size)
    plt.tick_params(axis='x', labelsize=tick_label_size)
    plt.tick_params(axis='y', labelsize=tick_label_size+2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=legend_size)
    plt.xticks(all_ke_values)


    save_path_ssim = os.path.join(save_dir, f'{target_codebook}_MS_SSIM.pdf')
    plt.savefig(save_path_ssim, format='pdf', bbox_inches='tight')
    print(f"MS-SSIM 图已保存: {save_path_ssim}")
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 设置数据根目录
    # 这里填写你 ls output/Ke 所在的那个绝对路径或相对路径
    root_directory = 'output/Ke' 

    # 2. 设置你要查询的条件
    my_snr_list = [10, 20]
    my_target_codebook = '16E' # 选择 '16E' 或 '64E'

    # 3. 设置保存位置
    my_save_dir = 'output/Ke'

    # 4. 运行绘图
    plot_ke_performance(root_directory, my_snr_list, my_target_codebook, my_save_dir)
    
    # 如果想画 64E 的，可以再调一次：
    # plot_ke_performance(root_directory, 18, '64E', my_save_dir)