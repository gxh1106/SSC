import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 读取两个方案的 TensorBoard CSV 数据
# 方案 1 (原本的方案)
df1 = pd.read_csv(r'C:\Users\18228\Desktop\loss_16xD_train_VQ_bpp2_32C_65536E.csv')

# 方案 2
df2 = pd.read_csv(r'C:\Users\18228\Desktop\loss_16xD_train_SSC_bbp2_32C_16E_4D.csv')

smooth_weight = 0.8  # 平滑权重，越接近 1 越平滑
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

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# 4. 绘制 Scheme 1
ax.plot(df1['Step'], df1['Value'], color='#1f77b4', alpha=0.08, linewidth=0.6) # 极细极淡的毛刺
ax.plot(df1['Step'], df1['Smoothed'], 
        color='#1f77b4', linestyle='--', linewidth=1.0,
        label='sDAC')

# 5. 绘制 Scheme 2
ax.plot(df2['Step'], df2['Value'], color='#d62728', alpha=0.08, linewidth=0.6)
ax.plot(df2['Step'], df2['Smoothed'], 
        color='#d62728', linestyle='-', linewidth=1.0, 
        label='Proposed SSC')

# 6. 【核心修改：优化坐标轴显示范围，防止切边】
ax.set_ylim([0.0, 0.017])       # 👈 将上限从 0.03 稍微提高到 0.032，防止顶部的 0.030 标签和起点圆点被切
ax.set_xlim([-5000, 205000])    # 👈 让 X 轴左右两端各有一点点富余，保证两端的实验数据和标签不紧贴边缘

# 7. 轴标签与网格美化
ax.set_xlabel('Training Step', fontsize=13, fontweight='bold', labelpad=10) # 增大间距
ax.set_ylabel('Loss Value', fontsize=13, fontweight='bold', labelpad=10)    # 增大间距
ax.tick_params(axis='both', labelsize=11)

ax.grid(True, linestyle=':', alpha=0.5, color='#b0b0b0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 8. 图例
ax.legend(loc='upper right', fontsize=14, edgecolor='black', framealpha=0.8)

# 9. 紧凑排版并保存
plt.tight_layout()

# 同时保存为高清 PNG 和 矢量 PDF 格式（完美解决切边问题）
plt.savefig('loss_comparison.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('zoomed_stable_loss_comparison.png', format='png', bbox_inches='tight')

plt.show()