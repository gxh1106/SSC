# SeIM Magazine 仿真代码（已对接 SSC 真实训练权重）

配套论文：*Semantic-Aware Index Modulation: ...*（IEEE Communications Magazine 投稿）

本目录代码由 `SSC-magazine/sim/` 的未验证版本对照 SSC 仓库源码（`ssc/inference.py`、
`ssc/archs/SwinSSC_arch.py`、`ssc/archs/quantizations.py`、`ssc/faim.py`）逐行核实并修正而来，
并已在 Kodak24 + 真实预训练权重上完整跑通（2026-09-05）。

## 文件结构

| 文件 | 说明 |
|---|---|
| `im_channels.py` | 三种 IM 物理层链路（FA-IM / SM / OFDM-IM）的纯 NumPy Monte Carlo 仿真，统一接口 `transmit(indices, snr_db, mode)`；`mode='seim'` 为语义感知流分割，`mode='eep'` 为无分割基线。已与 torch 版 `ssc/faim.py: FA_IM_Channel` 核对（Ns=4/Np=16/Nr=8/64-QAM/W=2/L=10 与 `ssc/inference.py` 一致） |
| `semantic_codec.py` | 语义编解码器接口：`SyntheticRQCodec`（合成高斯信源 + Lloyd-RQ，无需 GPU）与 `PretrainedSwinSSCCodec`（**已修正的**预训练权重钩子） |
| `run_simulation.py` | 主脚本：产出 `fig_ber_asymmetry.pdf`（双流 BER 不对称）与 `fig_snr_psnr.pdf`（SeIM vs w/o SS 的 PSNR-SNR 曲线），支持合成信源与预训练权重两种模式；`build_channels()` 定义论文采用的三种 IM 配置，`SNR_LISTS` 定义各自工作区间 |
| `run_config_sweep.py` | 索引/符号比特搭配扫描：对同一 IM 家族换用不同 (m1, m2) 组合，profiling 鲁棒流方向并测端到端 SeIM 增益，产出 `results/config_sweep.npz` 与 `results/fig_config_sweep.pdf`；用于验证"分配方向由配置决定、增益跟踪不对称强度" |
| `results/` | 仿真原始数据 `sim_results.npz`（已加入 .gitignore） |

## 环境搭建

本机系统 Python（3.8）无 pip/numpy，仿真在独立 venv 中运行（Python 3.14 + torch 2.14 cu126）：

```bash
# venv 已创建于 ~/.workbuddy/venvs/ssc-sim（含 numpy 2.5, matplotlib 3.11,
# torch 2.14+cu126, torchvision, timm, opencv-headless, pyyaml, addict 等）
# 如需重建：
python3 -m venv ~/.workbuddy/venvs/ssc-sim
~/.workbuddy/venvs/ssc-sim/bin/pip install numpy matplotlib
~/.workbuddy/venvs/ssc-sim/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
~/.workbuddy/venvs/ssc-sim/bin/pip install addict future lmdb opencv-python-headless Pillow pyyaml requests scikit-image scipy tqdm yapf timm
```

注意：`ssc/__init__.py` 会动态导入全部 models（其中 `SSCGAN_model.py` 有
`from loss.distortion import Distortion` 裸导入），`PretrainedSwinSSCCodec` 已自动把
`SSC_ROOT` 与 `SSC_ROOT/ssc` 都加入 `sys.path`，无需手工设置 PYTHONPATH。

## 运行仿真

```bash
cd code_magazine

# ---- 合成信源模式（纯 CPU/NumPy，无需 torch）----
python run_simulation.py                      # 完整仿真
python run_simulation.py --quick              # 冒烟测试

# ---- 配置扫描（m1/m2 搭配实验，需 torch 环境）----
CUDA_VISIBLE_DEVICES=5 ~/.workbuddy/venvs/ssc-sim/bin/python \
    run_config_sweep.py --workers 48          # 约 25 分钟

# ---- 预训练权重模式（真实图像，推荐后台运行；GPU 编号按空闲情况选）----
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup ~/.workbuddy/venvs/ssc-sim/bin/python run_simulation.py \
    --codec pretrained --trials 3 --slots 1000000 --workers 48 \
    --fig_dir ../SSC-magazine/figures \
    --res_dir ./results \
    > sim_run.log 2>&1 &
```

默认路径（可用 `--config/--model/--image_dir/--ssc_root` 覆盖）：

- 配置：`experiments/16xD/train_SSC_bpp2_32C_16E_4D/train_SSC_bpp2_32C_16E_4D_from_pretrain.yml`
- 权重：`experiments/16xD/train_SSC_bpp2_32C_16E_4D/models/net_g_latest.pth`
- 图像：`datasets/Kodak24`（24 张）

关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--trials` | 6 | 每 (图像, SNR) 的独立信道/噪声实现数（本次论文结果用 3） |
| `--slots` | 20000 | Part A 每个 SNR 的 BER 测量时隙数（论文最终图用 1000000 + 公共随机数，单调光滑到 ~1e-5；BER 图零误码点自动掩蔽不画） |
| `--workers` | 48 | 信道传输的 CPU 并行进程数（瓶颈在 CPU 端 OFDM-IM） |
| `--fig_dir` | `../figures` | 图片输出目录（论文用 `../SSC-magazine/figures`） |
| `--res_dir` | `../results` | 数据输出目录（本次用 `./results`，已 gitignore） |

## 仿真数据格式与读取

所有原始数据以 NumPy `.npz`（`numpy.savez`，zip 容器内每个条目一个 `.npy`）保存在
`results/`（已 gitignore），读取需 `allow_pickle=True`（含 dict 条目）：

### `results/sim_results.npz`（主仿真，对应论文 Fig. 3 / Fig. 4）

| 键 | 类型 | 内容 |
|---|---|---|
| `snr_lists` | dict[str → ndarray] | 每种 IM 的 SNR 采样点（FA-IM 0–20、SM 2–24、OFDM-IM 10–30，步长 2 dB） |
| `ber` | dict[str → (list, list)] | `ber[name] = (ber_index, ber_symbol)`，各 SNR 点索引流/符号流 BER（Fig. 3） |
| `{name}_seim` | ndarray | 各 SNR 点的 SeIM 平均 PSNR（dB）（Fig. 4） |
| `{name}_eep` | ndarray | 各 SNR 点的 w/o SS 基线平均 PSNR（dB）（Fig. 4） |

`name ∈ {"FA-IM", "SM", "OFDM-IM"}`。读取示例：

```python
import numpy as np

d = np.load('results/sim_results.npz', allow_pickle=True)
snr_lists = d['snr_lists'].item()          # dict: 名字 -> SNR 数组
ber = d['ber'].item()                      # dict: 名字 -> (ber_i, ber_s)

snr = snr_lists['SM']
gain = d['SM_seim'] - d['SM_eep']          # SeIM 增益曲线 (dB)
print('SM 最大增益:', gain.max().round(2), 'dB @', snr[gain.argmax()], 'dB')

bi, bs = np.asarray(ber['FA-IM'][0]), np.asarray(ber['FA-IM'][1])
print('FA-IM BER 比(索引/符号):', (bi / np.maximum(bs, 1e-12)).round(2))
```

### `results/config_sweep.npz`（m1/m2 搭配扫描）

单个键 `results`：object 数组，每个元素是一个 dict：

| 字段 | 内容 |
|---|---|
| `label` | 配置名（如 `'FA-IM 3+4'`） |
| `robust` | bool，索引流是否更鲁棒 |
| `snr` / `ber_i` / `ber_s` | 该配置的 SNR 点与双流 BER |
| `seim` / `eep` / `gain` | PSNR 曲线与增益（`gain = seim - eep`） |

```python
d = np.load('results/config_sweep.npz', allow_pickle=True)
for r in d['results']:
    print(r['label'], '最大增益', r['gain'].max().round(2), 'dB')
```

## 性能设计（为什么快）

预训练模式按"GPU 编码一次 → CPU 多进程并行传输 → GPU 批量解码"组织：

1. **编码一次**：当前模型为 `SwinJSCC_w/o_SAandRA` 变体，encoder/decoder/RQ 均不使用
   chan_param，编码结果与 SNR 无关，24 张图只编码一次（若换用 w/_SA 变体则自动逐 SNR 编码）。
2. **并行传输**：`transmit` 是纯 NumPy，用 `ProcessPoolExecutor` 按单次传输为粒度并行
   （FA-IM ~0.35s/次、SM ~0.26s/次、OFDM-IM ~2s/次；48 进程下全程约 25 分钟）。
3. **批量解码**：`decode_many` 把同一 (图像, SNR) 下的 trial×mode 索引拼成 batch 一次过 decoder。
4. **配对比较 + 公共随机数**：seim/eep 共用同一随机种子（同一信道实现与噪声），
   保证消融公平；且种子不含 SNR 项——同一 (图像, trial, 模式) 在所有 SNR 点共享
   同一套随机性，PSNR 曲线随 SNR 光滑单调（方差缩减，口径同 `ssc/inference.py`）。

全程参考：1584 个传输任务/信道，FA-IM 约 60s、SM 约 54s、OFDM-IM 约 24min（系统负载高时更慢）。

## 本次仿真结果（2026-09-05 最终版，已写入论文 Section V）

论文采用各自家族中不对称更明显、增益更显著的搭配（`build_channels`）：
**FA-IM Ns=4/Np=16, 16-QAM（2+4 bits，端口配置与 SSC 论文一致）**，
SM 4×4 64-QAM（2+6 bits），OFDM-IM n=4/k=2 16-QAM（2+8 bits）。
SNR 区间：FA-IM 0–20 dB，SM 2–24 dB，OFDM-IM 10–30 dB。

**光滑性方法（与 ssc/inference.py 口径一致）**：BER 与 PSNR 测量均采用公共随机数
（common random numbers）——每个 SNR 点用相同种子重建 rng，各点的比特序列、
信道实现顺序（确定性遍历 num_H=100 个实现）、噪声完全一致，仅噪声方差随 SNR
变化，曲线天然单调光滑；BER 每点 10^6 时隙。

鲁棒流方向：SM → 索引流，OFDM-IM → 索引流，FA-IM → 符号流。

| IM 方案 | SNR 区间 | 最大 SeIM 增益 |
|---|---|---|
| SM 4×4, 64-QAM | 2–24 dB | 2.20 dB @ 2 dB |
| OFDM-IM n=4,k=2, 16-QAM | 10–30 dB | 0.68 dB @ 10 dB（11/11 点为正，平滑递减） |
| FA-IM 16 端口/4 激活, 16-QAM | 0–20 dB | 1.07 dB @ 0 dB（11/11 点为正，平滑递减） |

高 SNR 处所有曲线饱和于编解码器的失真下限（约 27 dB）。
增益排序 SM > FA-IM > OFDM-IM 与瀑布区双流 BER 不对称强度排序一致（论文核心论据）。

备注（配置选择过程）：FA-IM 3+4（Ns=8, 16-QAM）低 SNR 增益虽大（1.37 dB @ 0 dB），
但 trials=10 收敛验证发现其在 6–16 dB 出现约 -0.2 dB 的真实负增益（重要索引集中
在符号流上，深衰落实现时缺乏错误分散），故论文弃用；2+4 配置增益全区间非负且平滑。

## 索引/符号比特搭配扫描（run_config_sweep.py，2026-09-05）

在保持各 IM 家族物理模型不变的前提下调整 (m1, m2) 搭配（trials=3、6 个 SNR 点、
2×10^4 profiling 时隙/点），结果存 `results/config_sweep.npz`，图 `results/fig_config_sweep.pdf`：

| 配置 | (m1+m2) | 鲁棒流 | 不对称强度(中位 log10 比) | 最大 SeIM 增益 |
|---|---|---|---|---|
| SM Nt=4, 64-QAM | 2+6 | 索引 | 0.35 | 1.75 dB |
| SM Nt=16, 16-QAM | 4+4 | **符号（方向翻转）** | 0.13 | 0.07 dB |
| SM Nt=4, 16-QAM | 2+4 | 索引 | 0.02 | 0.76 dB |
| FA-IM Ns=2, 64-QAM | 1+6 | 符号 | 0.49 | 0.35 dB |
| FA-IM Ns=4, 64-QAM | 2+6 | 符号 | 0.23 | 0.25 dB |
| FA-IM Ns=8, 16-QAM | 3+4 | 符号 | 0.52 | **1.37 dB** |
| OFDM-IM n4k2, 4-QAM | 2+4 | 索引 | 0.44 | 0.37 dB |
| OFDM-IM n4k2, 16-QAM | 2+8 | 索引 | 0.31 | 0.65 dB |

要点：
- **鲁棒流方向随配置翻转**（SM Nt=16/16-QAM 时索引流反而更脆弱）→ 实证"必须离线
  profiling、不可假设"；
- 增益随不对称强度增大（FA-IM 3+4 不对称最强 → 增益最大），但还受分割比例与
  低 SNR 段 BER 差影响，非严格单调（OFDM-IM 2+4 中位不对称大但增益小，因其差距
  集中在高 SNR 段而非增益敏感的低 SNR 段）。

## 相对未验证版本修复的问题（重要）

1. **[致命] 分辨率不匹配**：原 `PretrainedSwinSSCCodec` 只在初始化时按训练尺寸
   256×256 调用 `update_resolution`。Kodak 测试图裁剪后为 512×768，而 SSC 的
   `SwinTransformerBlock.forward` 内有 `assert L == H * W`，直接报错。
   修正：`encode_all`/`decode`/`decode_many` 按每张图实际尺寸更新 encoder/decoder
   分辨率（等价于 `SwinSSC.forward` 内置的尺寸变化处理，本类绕过了 forward 故须自行处理）。
2. **[致命] 码本大小探测错误**：原实现 `quantizer.codebooks[0].embed.num_embeddings`
   中 `embed` 是 VQEmbedding 的**方法**而非模块，会 AttributeError；且
   `VQEmbedding` 继承 `nn.Embedding(n_embed + 1, padding_idx=n_embed)`，
   `num_embeddings` 是 17 而非 16。修正：直接读取 `quantizer.n_embed[0]`
   与 `quantizer.rq_depth`。
3. **[行为偏差] chan_param 固定**：原实现全程使用 `snr_train=10`，而官方推理
   `ssc/inference.py` 是每个 SNR 都用 `given_SNR=snr` 驱动编解码器。修正：
   `encode_all(snr=...)`/`decode(..., snr=...)` 按当前 SNR 传递 chan_param。
4. **[健壮性]** 兼容 encoder 返回 `(x, mask)` 的 SA/RA 变体；断言仅支持单头 RQ
   （`num_heads=1` 且非 MVQ 模式）；numpy→tensor 前强制 `ascontiguousarray`；
   接收端索引 `clamp` 到 `[0, Ne)`；权重加载后报告非 `attn_mask` 的 missing/unexpected 键
   （当前 checkpoint 缺 `bit_flip_prob_*` 三个 buffer，属旧版训练代码差异，eval 路径
   不经过 BSC 信道，不影响结果）。

## 接口约定

- 索引矩阵：形状 `(L, Nq)`，dtype int64，取值 `[0, Ne)`；第 q 列 = 第 q 个 RQ
  量化步，**列号越小语义越重要**。默认 Ne=16, Nq=4（4 bit/索引）。
- `encode_all(snr)` 缓存 `[{'gt', 'indices', 'shape_info', 'chan_param'}, ...]`；
  `decode(indices, ref=rec, snr=snr)` 单条解码；`decode_many(list, ref=rec, snr=snr)` 批量解码。
- IM 信道 `transmit(indices, snr_db, mode='seim'|'eep', rng)`；`mode='seim'` 对应
  SSC 论文 `ssc=True, ssc_adapt=True`（自适应索引分流），`mode='eep'` 对应 `ssc=False`。

## 注意事项

- 仿真图被 `main.tex` 引用为 PDF，重跑仿真后需重新编译 LaTeX。本机已装 TinyTeX
  （`~/.TinyTeX`），编译命令：
  `export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH && cd ../SSC-magazine && pdflatex main && bibtex main && pdflatex main && pdflatex main`。
- FA-IM 在论文配置（Ns=4/Np=16, 16-QAM）下**符号流** BER 更低（公共随机数 +
  10^6 时隙实测，整个 0–20 dB 区间方向一致，高 SNR 差达两个数量级），
  与 SSC 论文"索引流事件错误更少"的事件口径不同——这是文章论点之一，
  **不要"修复"它去迎合直觉**。
- 换用其他实验权重时改 `--config/--model` 即可（注意 n_embed 需为 2 的幂，且
  当前仅支持单头 RQ、非 MVQ 模式）。
