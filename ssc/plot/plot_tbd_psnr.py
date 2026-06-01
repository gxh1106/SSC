import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 读取两个方案的 TensorBoard PSNR CSV 数据 (请确保替换为您的 PSNR 数据路径)
# 方案 1 (sDAC)
df1 = pd.read_csv(r'C:\Users\18228\Desktop\psnr_16xD_train_VQ_bpp2_32C_65536E.csv') # 👈 记得更换为 PSNR 的 CSV 路径

# 方案 2 (Proposed SSC)
df2 = pd.read_csv(r'C:\Users\18228\Desktop\psnr_16xD_train_SSC_bbp2_32C_16E_4D.csv') # 👈 记得更换为 PSNR 的 CSV 路径

smooth_weight = 0.0  # 平滑权重，您的 PSNR 曲线在 80k-100k 处有剧烈波动，0.75 可以较好地保留真实的趋势

# 2. 指数移动平均平滑函数
def smooth_curve(values, weight=0.85):
    smoothed = []
    last = values[0]
    for val in values:
        if np.isnan(val):
            val = last
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

df1['Smoothed'] = smooth_curve(df1['Value'].tolist(), weight=smooth_weight)
df2['Smoothed'] = smooth_curve(df2['Value'].tolist(), weight=smooth_weight)

# 3. 学术配置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# 使用稍微宽一点的画布，配合手动边距调整，彻底解决部分环境下标签切边的问题
fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

# 4. 绘制 sDAC (使用原本的蓝色调)
ax.plot(df1['Step'], df1['Value'], color='#1f77b4', alpha=0.08, linewidth=0.6) # 背景浅色毛刺
ax.plot(df1['Step'], df1['Smoothed'], 
        color='#1f77b4', linestyle='-', linewidth=1.5,
        marker='o', markersize=4, markevery=5, 
        label='sDAC')

# 5. 绘制 Proposed SSC (使用原本的红色调，对应截图里大幅涨点的曲线)
ax.plot(df2['Step'], df2['Value'], color='#d62728', alpha=0.08, linewidth=0.6) # 背景浅色毛刺
ax.plot(df2['Step'], df2['Smoothed'], 
        color='#d62728', linestyle='-', linewidth=1.5, 
        marker='s', markersize=4, markevery=5, 
        label='Proposed SSC')

# 6. 【核心修改：根据 PSNR 数据调整坐标轴范围】
ax.set_ylim([22.0, 29.5])       # 👈 根据截图，PSNR 最低在 22.5 左右，最高接近 29，留出上下 margin
ax.set_xlim([-5000, 205000])    # 👈 X 轴两端留出富余，防止 0 和 200k 处的标点紧贴边缘被切

# 7. 轴标签与网格美化
ax.set_xlabel('Training Step', fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold', labelpad=8) # 👈 修改为 PSNR 标准学术标签
ax.tick_params(axis='both', labelsize=11)

# 将 X 轴的刻度显示规范化（例如 0, 25000, 50000... 保持和您之前的图一致）
# 如果想要像 TensorBoard 那样显示 20k, 40k，可以取消下面这两行的注释：
# ax.set_xticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000])
# ax.set_xticklabels(['0', '20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k'])

ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 8. 图例 (PSNR 是右上角高，如果图例在右上角挡住曲线，可以改成 loc='lower right')
ax.legend(loc='lower right', fontsize=14, edgecolor='black', framealpha=0.8)

# 9. 强制手动预留边距（比 tight_layout 更稳妥地防止 X、Y 轴标签被切边）
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

# 10. 保存为高清 PNG 和 矢量 PDF 格式
plt.savefig('psnr_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('psnr_comparison.png', format='png', bbox_inches='tight')

plt.show()