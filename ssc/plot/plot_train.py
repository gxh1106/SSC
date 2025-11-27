import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_specific_strategy(root_dir, target_c, target_d, save_dir='.'):
    """
    根据指定的 C (如 '32C') 和 D (如 '2D')，
    直接查找对应的 newTrain 和 oldTrain 文件夹并绘制对比图。
    """
    
    # 1. 构建文件夹名称
    # 假设命名格式严格为: "32C_2D_newTrain" 和 "32C_2D_oldTrain"
    config_prefix = f"{target_c}_{target_d}"
    
    folder_name_new = f"{config_prefix}_newTrain"
    folder_name_old = f"{config_prefix}_oldTrain"
    
    path_new = os.path.join(root_dir, folder_name_new)
    path_old = os.path.join(root_dir, folder_name_old)
    
    # 2. 读取数据辅助函数
    def read_data(folder_path):
        csv_path = os.path.join(folder_path, 'psnr_results.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns] # 去空格
                df = df.sort_values(by='SNR (dB)')           # 按SNR排序
                return df
            except Exception as e:
                print(f"读取失败 {csv_path}: {e}")
        return None

    df_new = read_data(path_new)
    df_old = read_data(path_old)

    # 如果两个文件都没找到，直接退出
    if df_new is None and df_old is None:
        print("错误: 未找到该配置下的 newTrain 或 oldTrain 数据，请检查路径。")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # 全局绘图样式
    plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2.5, 'lines.markersize': 9})

    # ==========================================
    # 1. 绘制 PSNR 对比图
    # ==========================================
    plt.figure(figsize=(6, 6))
    
    # 先画 Old (基准，蓝色虚线方块)
    if df_old is not None:
        plt.plot(df_old['SNR (dB)'], df_old['PSNR (dB)'], 
                 marker='s', linestyle='-', color='#1f77b4', 
                 label='w/o Stage 1')
    else:
        print("提示: 未找到 Old Strategy 数据")

    # 后画 New (重点，红色实线圆圈)
    if df_new is not None:
        plt.plot(df_new['SNR (dB)'], df_new['PSNR (dB)'], 
                 marker='o', linestyle='-', color='#d62728', 
                 zorder=10, # 保证覆盖在上面
                 label='Proposed')
    else:
        print("提示: 未找到 New Strategy 数据")

    plt.xlabel('SNR (dB)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14, loc='lower right')
    
    save_name_psnr = f"{config_prefix}_PSNR.pdf"
    plt.savefig(os.path.join(save_dir, save_name_psnr), format='pdf', bbox_inches='tight')
    print(f"  -> PSNR 图已保存: {save_name_psnr}")
    plt.show()

    # ==========================================
    # 2. 绘制 MS-SSIM 对比图
    # ==========================================
    plt.figure(figsize=(6, 6))
    
    if df_old is not None:
        plt.plot(df_old['SNR (dB)'], df_old['MS-SSIM'], 
                 marker='s', linestyle='-', color='#1f77b4', 
                 label='w/o Stage 1')

    if df_new is not None:
        plt.plot(df_new['SNR (dB)'], df_new['MS-SSIM'], 
                 marker='o', linestyle='-', color='#d62728', 
                 zorder=10,
                 label='Proposed')

    plt.xlabel('SNR (dB)', fontsize=16)
    plt.ylabel('MS-SSIM', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14, loc='lower right')
    
    save_name_ssim = f"{config_prefix}_MS_SSIM.pdf"
    plt.savefig(os.path.join(save_dir, save_name_ssim), format='pdf', bbox_inches='tight')
    print(f"  -> MS-SSIM 图已保存: {save_name_ssim}")
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 你的数据根目录
    root_directory = 'output/diff_train'
    
    # 2. 结果保存目录
    save_directory = 'output/diff_train'
    
    # ==========================================
    # 3. 在这里指定你想画的配置
    # ==========================================
    
    # 示例 1: 画 96C, 4D 的对比
    target_channel = '32C'  # 对应文件名里的 96C
    target_dim = '4D'       # 对应文件名里的 4D
    
    plot_specific_strategy(root_directory, target_channel, target_dim, save_directory)

    # 示例 2: 如果想接着画 32C, 2D，可以再调一次函数
    # plot_specific_strategy(root_directory, '32C', '2D', save_directory)