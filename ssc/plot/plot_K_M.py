import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_faim_performance(root_dir, target_k, target_m, save_dir='.', max_snr=20):
    """
    根据指定的 K 和 M 值 (例如 K=2, M=16)，
    定位到 '2K_16M' 文件夹，并绘制其中包含的所有方法的对比曲线。
    """
    
    # 1. 构建目标子文件夹路径
    # 文件夹命名规则看起来是: "{K}K_{M}M" (例如 2K_16M)
    subfolder_name = f"{target_k}K_{target_m}M"
    base_path = os.path.join(root_dir, subfolder_name)
    
    
    if not os.path.exists(base_path):
        print(f"错误: 找不到路径 '{base_path}'，请检查 K 和 M 值是否正确。")
        return

    # 2. 定义要绘制的方法及其样式配置 (字典列表)
    # 这样可以精确控制每条线的颜色、形状和层级
    methods_config = [
        
        {
            'folder': 'SSC',
            'label': 'SSC',
            'color': '#d62728',      # 蓝色 (Baseline)
            'marker': 'o',           # 圆圈
            'linestyle': '-',        # 实线
            'zorder': 10
        },
        {
            'folder': 'SSC_woIM',
            'label': 'SSC w/o IM',
            'color': '#ff7f0e',      # 橙色
            'marker': '^',           # 方块
            'linestyle': '-',       # 虚线 (消融实验)
            'zorder': 3
        },
        {
            'folder': 'SSC_woSS',
            'label': 'SSC w/o SS',
            'color': '#2ca02c',      # 绿色
            'marker': 'D',           # 三角
            'linestyle': '-',       # 虚线
            'zorder': 3
        },
        {
            'folder': 'sDAC_FA-IM',  # 文件夹名
            'label': 'sDAC with FA-IM', # 图例名
            'color': '#1f77b4',      # 红色 (重点推荐)
            'marker': 's',           # 星号
            'linestyle': '-',        # 实线
            'zorder': 5             # 画在最上层
        },
    ]

    os.makedirs(save_dir, exist_ok=True)
    
    # 全局绘图样式
    plt.rcParams.update({
        'font.size': 16, 
        'lines.linewidth': 2, 
        'lines.markersize': 8
    })

    # ==========================================
    # 1. 绘制 PSNR 对比图
    # ==========================================
    plt.figure(figsize=(6, 6))
    
    data_found = False # 标记是否至少找到一个文件

    for config in methods_config:
        method_folder = config['folder']
        csv_path = os.path.join(base_path, method_folder, 'psnr_results.csv')
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # 清理列名并排序
                df.columns = [c.strip() for c in df.columns]
                df = df[df['SNR (dB)'] <= max_snr]
                df = df.sort_values(by='SNR (dB)')
                
                # 绘图
                plt.plot(df['SNR (dB)'], df['PSNR (dB)'],
                         label=config['label'],
                         color=config['color'],
                         marker=config['marker'],
                         linestyle=config['linestyle'],
                         zorder=config['zorder'])
                data_found = True
            except Exception as e:
                print(f"读取失败 {method_folder}: {e}")
        else:
            # 如果某个方法在这个 K/M 下不存在，打印警告但不报错
            print(f"警告: 在 {subfolder_name} 下未找到 {method_folder}")

    if not data_found:
        print("未找到任何有效数据，跳过绘图。")
        plt.close()
        return

    # 设置图表细节
    plt.xlabel('SNR (dB)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=16, loc='lower right') # 手动指定位置以避免遮挡
    
    # 保存 PSNR
    save_name_psnr = f"{target_k}K_{target_m}M_PSNR.pdf"
    plt.savefig(os.path.join(save_dir, save_name_psnr), format='pdf', bbox_inches='tight')
    print(f"  -> PSNR 图已保存: {save_name_psnr}")
    plt.show()

    # ==========================================
    # 2. 绘制 MS-SSIM 对比图
    # ==========================================
    plt.figure(figsize=(6, 6))

    for config in methods_config:
        method_folder = config['folder']
        csv_path = os.path.join(base_path, method_folder, 'psnr_results.csv')
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                df = df[df['SNR (dB)'] <= max_snr]
                df = df.sort_values(by='SNR (dB)')
                
                plt.plot(df['SNR (dB)'], df['MS-SSIM'],
                         label=config['label'],
                         color=config['color'],
                         marker=config['marker'],
                         linestyle=config['linestyle'],
                         zorder=config['zorder'])
            except:
                pass

    plt.xlabel('SNR (dB)', fontsize=16)
    plt.ylabel('MS-SSIM', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=16, loc='lower right') # 手动指定位置以避免遮挡
    
    # 保存 MS-SSIM
    save_name_ssim = f"{target_k}K_{target_m}M_MS_SSIM.pdf"
    plt.savefig(os.path.join(save_dir, save_name_ssim), format='pdf', bbox_inches='tight')
    print(f"  -> MS-SSIM 图已保存: {save_name_ssim}")
    plt.show()

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 你的数据根目录
    root_directory = 'output/FA_IM'
    
    # 2. 结果保存目录
    save_directory = 'output/FA_IM'
    
    # ==========================================
    # 3. 指定参数
    # ==========================================
    target_k=2
    target_m=256

    max_snr=20
    # 场景 A: 绘制 2K, 16M 的结果
    plot_faim_performance(root_directory, target_k=target_k, target_m=target_m, save_dir=save_directory, max_snr=max_snr)

    # 场景 B: 绘制 4K, 256M 的结果 (取消注释即可运行)
    # plot_faim_performance(root_directory, target_k=4, target_m=256, save_dir=save_directory)