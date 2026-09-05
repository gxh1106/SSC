# SeIM Magazine 仿真对接与论文迭代——会话工作总结（2026-09-05）

> 本文档记录本次会话的全部工作内容与关键要点，供后续接续工作参考。
> 代码细节见同目录 `README.md`。

## 1. 任务背景

`SSC-magazine/` 中的 magazine 论文（SeIM，IEEE ComMag 投稿）此前配套的仿真代码
（`SSC-magazine/sim/`）从未加载真实网络权重验证过。本次任务：对照 SSC 仓库真实代码核查、
用 `experiments/` 中的训练权重实际仿真、按结果迭代论文，并探索 (m1, m2) 比特搭配。

## 2. 代码核查发现的 Bug（原 sim/semantic_codec.py，已全部修复）

修正后的代码在 `code_magazine/`：

1. **[致命] 分辨率不匹配**：原 `PretrainedSwinSSCCodec` 只在初始化时按训练尺寸 256×256
   `update_resolution`，Kodak 测试图（裁剪后 512×768）触发 `SwinTransformerBlock` 内
   `assert L == H * W` 报错。修复：encode/decode 前按实际图像尺寸更新 encoder/decoder 分辨率。
2. **[致命] 码本大小探测错误**：`codebooks[0].embed.num_embeddings` 中 `embed` 是
   `VQEmbedding` 的方法而非模块；且其继承 `nn.Embedding(n_embed+1, padding_idx=n_embed)`，
   `num_embeddings` 为 17 而非 16。修复：直接读 `quantizer.n_embed[0]` / `quantizer.rq_depth`。
3. **[行为偏差] chan_param 固定**：官方 `ssc/inference.py` 逐 SNR 传 `given_SNR`，原实现固定 10。
   修复：`encode_all(snr=...)` / `decode(..., snr=...)` 按当前 SNR 传递。
4. **[健壮性]** 兼容 SA/RA 变体的 `(x, mask)` 返回；断言单头 RQ 非 MVQ；`ascontiguousarray`；
   索引 clamp；权重加载报告非 attn_mask 的不匹配键。

核对一致未动的部分：`ad`/`embed`/`da` 调用链与 `SwinSSC_arch.forward_faim` 吻合；
numpy 版 IM 信道与 torch 版 `ssc/faim.py` 参数/算法一致。

## 3. 环境（本机新建）

- 系统 Python 3.8 无 pip/numpy → 安装独立 Python 3.14 并建 venv：`~/.workbuddy/venvs/ssc-sim`
  （numpy 2.5.2、matplotlib 3.11.1、torch 2.14.0+cu126、timm、opencv-headless、addict、pyyaml 等）
- LaTeX：安装 TinyTeX 至 `~/.TinyTeX`（含 ieeetran 等宏包），编译命令见 README
- GPU：使用空闲的 5 号卡（`CUDA_VISIBLE_DEVICES=5`）
- 仿真数据与日志已加入 `.gitignore`：`code_magazine/results/`、`code_magazine/sim_run*.log`

## 4. 仿真执行（三轮 + 一次扫描）

模型：`experiments/16xD/train_SSC_bpp2_32C_16E_4D/models/net_g_latest.pth`
（C=32、embed_dim=4、n_embed=16、rq_depth=4，恰好匹配 IM 信道 4 bit/索引约定）；
图像：Kodak24 全部 24 张；trials=3；seim/eep 配对比较（同种子同信道实现）。

性能设计：GPU 编码一次（`w/o_SAandRA` 变体与 SNR 无关）→ 48 进程并行 NumPy 信道传输
→ GPU 批量解码（`decode_many`）。全程约 25 分钟（瓶颈在 CPU 端 OFDM-IM ML 检测）。

### 关键结果（论文采用，第三轮）

| IM 方案（论文配置） | SNR 区间 | 鲁棒流 | 最大 SeIM 增益 |
|---|---|---|---|
| SM 4×4, 64-QAM（2+6 bits） | 2–24 dB | 索引流 | 2.04 dB @ 4 dB |
| OFDM-IM n4k2, 16-QAM（2+8 bits） | 10–30 dB | 索引流 | 0.65 dB @ 10 dB |
| FA-IM Ns=8/Np=16, 16-QAM（3+4 bits） | 0–20 dB | 符号流 | 1.37 dB @ 0 dB |

BER 图最终用 10⁶ 时隙/点（均分到全部信道实现），零误码点掩蔽，曲线光滑到 ~1e-5。

### 配置扫描（run_config_sweep.py，8 个配置）

- SM (16 天线, 16-QAM, 4+4)：**鲁棒流方向翻转**为符号流，增益塌缩到 0.07 dB；
- FA-IM (Ns=4, 64-QAM, 2+6)：增益仅 0.25–0.41 dB（双流接近等可靠）；
- 换成 3+4 后不对称拉大到 ~4 倍 → 增益 1.37 dB（论文改用它）；
- 结论：增益跟踪不对称强度 × 低 SNR 段差距 × 分割比例，非严格单调。

### 为什么 BER 差距大、PSNR 增益小（用户疑问的解答）

对数轴放大倍数差；绝对错误比特很少被海量正确索引稀释；RQ 深层索引本就不敏感；
SeIM 是置换不是编码，增益上限 = BER 差 × 重要层敏感度；高 SNR 饱和于 codec 失真下限 ~27 dB。

## 5. 论文 main.tex 修改清单

1. **Section V Setup**：合成信源 → 真实预训练 codec（[ref-SSC] 的 SwinSSC，RQ 4 步 × 16 码本）
   + Kodak24；分链路 SNR 区间；24 图 × 3 实现配对比较；profiling 2×10⁵ 时隙；
   FA-IM 配置改为 16 端口/8 激活/16-QAM（"in the spirit of [ref-SSC]"）。
2. **Table II**：SM 2.0 / OFDM-IM 0.7 / FA-IM 1.4 dB；鲁棒流 Index/Index/Symbol。
3. **Section IV Profile 步骤**：新增"分配方向非方案家族固定属性，调制参数一变必须重新实测"。
4. **Section V 新增配置扫描段**：SM 4+4 方向翻转（2.0→<0.1 dB）、FA-IM 2+6→3+4
   （BER 差 1.7 倍→近 4 倍，增益 0.4→1.4 dB）；提炼两条设计规则（方向只能实测；
   增益随不对称增长 → 调制与语义分割应协同设计）。
5. **Section VI-B**：呼应扫描发现（IM 参数重塑不对称本身，含方向）。
6. **摘要**：案例研究改为"真实图像 + 预训练 codec"。
7. **Fig. 2（fig-seim-arch）**：字号 \small→\scriptsize、方框加宽（17/19mm）消除文字溢出；
   分支箭头改 `(split.north) |- (idxmap.west)` 直角走线对齐；流标签居中。
8. 结果讨论逐方案重写：SM 高端增益归零、OFDM-IM 向 30 dB 平滑递减、
   增益排序 SM > FA-IM > OFDM-IM 与不对称强度排序一致。

编译：TinyTeX 全链 `pdflatex → bibtex → pdflatex → pdflatex` 通过，**7 页、0 错误、
引用全部解析**，逐页渲染检查过 Fig. 2/3/4 与 Table II。

## 6. 产物位置

| 内容 | 路径 |
|---|---|
| 修正后的仿真代码 | `code_magazine/{im_channels,semantic_codec,run_simulation,run_config_sweep}.py` |
| 主仿真数据 | `code_magazine/results/sim_results.npz`（gitignored） |
| 扫描数据/图 | `code_magazine/results/config_sweep.npz`、`fig_config_sweep.pdf/.png` |
| 论文用图 | `SSC-magazine/figures/fig_{ber_asymmetry,snr_psnr}.{pdf,png}` |
| 论文 PDF | `SSC-magazine/main.pdf`（7 页） |
| 使用文档 | `code_magazine/README.md`（含 npz 数据格式与读取示例） |
| 运行日志 | `code_magazine/sim_run*.log`、`sweep_run.log`、`ber_rerun.log`（gitignored） |

## 7. 遗留事项（见 SSC-magazine/AGENT_BRIEFING.md 第 5 节）

1. 文献 21 篇仍超 ComMag 上限 15 篇，需删减并同步清理 `\cite`；
2. 页数当前正好 7 页（本次新增扫描段落后仍需留意）；
3. 投稿前在致谢披露 AI 辅助使用（main.tex 中有 TODO 注释）；
4. 录用阶段补 ≤150 词 bio + ORCID。
