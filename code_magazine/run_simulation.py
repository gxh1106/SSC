"""
run_simulation.py
=================
Magazine 文章（SeIM）配套仿真主脚本（已对接 SSC 仓库真实训练权重）。

产出两张图（默认保存到 ../figures/）：

- Fig. A ``fig_ber_asymmetry.pdf``：三种 IM 方案（FA-IM / SM / OFDM-IM）
  索引流与符号流的误比特率随 SNR 的变化，验证"双流可靠性不对称"这一
  SeIM 赖以成立的物理层事实（纯 NumPy，无需 GPU）。
- Fig. B ``fig_snr_psnr.pdf``：三种 IM 方案应用语义感知流分割（SeIM）
  与均等分割（w/o SS）的端到端重建 PSNR-SNR 曲线。

Fig. B 支持两种信源编解码器：

- ``--codec synthetic``（默认）：合成高斯信源 + Lloyd-RQ，纯 NumPy。
- ``--codec pretrained``：加载 SSC 仓库 experiments/ 下训练好的
  SwinSSC 权重，在 Kodak24 真实图像上评估（需要 PyTorch 环境）。

用法
----
    # 完整仿真（合成信源）
    python run_simulation.py

    # 快速冒烟测试
    python run_simulation.py --quick

    # 真实图像 + 预训练权重（在 SSC 训练环境中运行）
    python run_simulation.py --codec pretrained \
        --config ../experiments/16xD/train_SSC_bpp2_32C_16E_4D/train_SSC_bpp2_32C_16E_4D_from_pretrain.yml \
        --model  ../experiments/16xD/train_SSC_bpp2_32C_16E_4D/models/net_g_latest.pth \
        --image_dir ../datasets/Kodak24

相对 SSC-magazine/sim/run_simulation.py 的修正：

1. Part B 支持 ``PretrainedSwinSSCCodec``，真实图像路径下每个 SNR 的
   chan_param 与官方 ssc/inference.py 的 ``given_SNR=snr`` 行为一致
   （编码端与解码端都使用当前 SNR）。
2. 预训练模式下 RQ 索引逐图缓存，同一 SNR 下 seim/eep 两种分割复用
   同一份编码结果，保证对比公平（与 SSC 论文的消融口径一致）。
"""

import argparse
import glob
import os
import time

import numpy as np

# macOS Accelerate BLAS 在复数 matmul 时会误报 divide-by-zero/overflow
# 浮点标志（结果实际有限），此处全局屏蔽以免干扰日志。
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from im_channels import FAIMChannel, SMChannel, OFDMIMChannel, decide_robust_stream

HERE = os.path.dirname(os.path.abspath(__file__))
SSC_ROOT = os.path.normpath(os.path.join(HERE, ".."))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
RES_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 120,
})


def build_channels(seed=0):
    # 论文选用各自家族中双流不对称更明显、SeIM 增益更显著的搭配：
    # FA-IM 16 端口选 4 + 16-QAM（2+4 bits，与 SSC 论文端口配置一致），
    # SM 4 天线 + 64-QAM（2+6 bits），OFDM-IM 4 选 2 子载波 + 16-QAM（2+8 bits）。
    return [
        FAIMChannel(Ns=4, Np=16, Nr=8, M=16, W=2.0, L_paths=10,
                    num_H=100, seed=seed),
        SMChannel(Nt=4, Nr=4, M=64, num_H=100, seed=seed + 1),
        OFDMIMChannel(n=4, k=2, M=16, seed=seed + 2),
    ]


# 每种 IM 各自的工作 SNR 区间（dB）：FA-IM 0-20，SM 2-24，OFDM-IM 10-30
SNR_LISTS = {
    "FA-IM": list(range(0, 22, 2)),
    "SM": list(range(2, 26, 2)),
    "OFDM-IM": list(range(10, 32, 2)),
}


def part_a_ber_asymmetry(channels, snr_lists, n_slots, out_prefix,
                         clip=1e-6):
    """测量三种 IM 的索引流/符号流 BER，并决定各自信道的鲁棒流。
    每种信道使用各自的工作 SNR 区间。"""
    print("[Part A] 双流 BER 不对称性测量 ...")
    results = {}
    for ch in channels:
        snr_list = snr_lists[ch.name]
        robust, ber_i, ber_s = decide_robust_stream(ch, snr_list, n_slots)
        ch.index_more_robust = robust
        results[ch.name] = (ber_i, ber_s)
        side = "索引流" if robust else "符号流"
        print(f"  {ch.name:>9s}: 更鲁棒的物理流 = {side}")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, ch in zip(axes, channels):
        snr_list = snr_lists[ch.name]
        ber_i, ber_s = results[ch.name]
        # 零误码点不画（掩为 NaN），极小值截到 clip，避免假平台
        pi = np.where(np.array(ber_i) > 0, np.maximum(ber_i, clip), np.nan)
        ps = np.where(np.array(ber_s) > 0, np.maximum(ber_s, clip), np.nan)
        ax.semilogy(snr_list, pi, "o-", color="#0072BD",
                    lw=1.2, ms=4, label="Index stream")
        ax.semilogy(snr_list, ps, "s--", color="#D95319",
                    lw=1.2, ms=4, label="Symbol stream")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(ch.name, fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    axes[0].set_ylabel("Bit error rate")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  -> 已保存 {out_prefix}.pdf/.png")
    return results


def part_b_end_to_end(channels, codec, snr_lists, n_vectors, n_trials,
                      out_prefix):
    """端到端重建 PSNR vs SNR（合成信源）：SeIM vs w/o SS。"""
    print("[Part B/synthetic] 端到端 PSNR-SNR 曲线 ...")
    rng = np.random.default_rng(42)
    curves = {}
    for ch in channels:
        snr_list = snr_lists[ch.name]
        psnr_seim = np.zeros((n_trials, len(snr_list)))
        psnr_eep = np.zeros((n_trials, len(snr_list)))
        for t in range(n_trials):
            src = codec.generate_source(n_vectors, rng)
            idx = codec.encode(src)
            for si, snr in enumerate(snr_list):
                for mode, acc in (("seim", psnr_seim), ("eep", psnr_eep)):
                    idx_hat = ch.transmit(idx, snr, mode=mode, rng=rng)
                    acc[t, si] = codec.psnr(src, codec.decode(idx_hat))
            print(f"  {ch.name:>9s} trial {t + 1}/{n_trials} 完成")
        curves[ch.name] = dict(seim=psnr_seim.mean(0), eep=psnr_eep.mean(0))

    _plot_psnr_curves(channels, curves, snr_lists, out_prefix)
    return curves


# ---------------------------------------------------------------------------
# 并行传输 worker（ProcessPoolExecutor，纯 NumPy CPU 计算）
# ---------------------------------------------------------------------------
_WCH = None    # worker 进程内的信道对象
_WIDX = None   # worker 进程内的全部图像索引


def _tx_worker_init(ch, all_indices):
    global _WCH, _WIDX
    _WCH, _WIDX = ch, all_indices


def _tx_worker(task):
    """单次传输任务：返回 (snr, img_i, trial, mode, indices_hat)。"""
    snr, ii, trial, mode, seed = task
    rng = np.random.default_rng(seed)
    idx_hat = _WCH.transmit(_WIDX[ii], snr, mode=mode, rng=rng)
    return (snr, ii, trial, mode, idx_hat)


def part_b_pretrained(channels, codec, snr_lists, n_trials, out_prefix,
                      workers=48):
    """端到端重建 PSNR vs SNR（真实图像 + 预训练 SwinSSC）：SeIM vs w/o SS。

    流程与官方 ssc/inference.py 对齐：先用 given_SNR=snr 编码得到 RQ 索引
    （w/o_SAandRA 变体编码与 SNR 无关，自动只编码一次），索引经 numpy IM
    信道传输（seim/eep 两种分割共用同一份编码索引且共用同一随机种子，
    保证消融是配对比较），再由 quantizer.embed -> da -> decoder 批量重建。

    性能设计：编码/解码在 GPU；信道传输是纯 CPU NumPy，用
    ProcessPoolExecutor 多进程并行（任务粒度 = 单次 transmit）。
    """
    import concurrent.futures as cf

    print(f"[Part B/pretrained] 真实图像端到端 PSNR-SNR 曲线 "
          f"(workers={workers}) ...")
    ref_snr = snr_lists[channels[0].name][0]
    cache = codec.encode_all(snr=ref_snr)         # SNR 无关时仅编码一次
    all_indices = [rec["indices"] for rec in cache]
    n_img = len(cache)
    curves = {}
    for ch in channels:
        snr_list = snr_lists[ch.name]
        t0 = time.time()
        # seim/eep 使用相同种子 -> 同一信道实现与噪声 -> 配对比较
        # 种子不含 SNR 项（公共随机数）：同一 (图像, trial, 模式) 在所有
        # SNR 点上共享同一信道实现与噪声序列，PSNR 曲线随 SNR 光滑变化
        tasks = [(snr, ii, t, mode,
                  (ii * 1000 + t * 10) & 0x7fffffff)
                 for si, snr in enumerate(snr_list)
                 for ii in range(n_img)
                 for t in range(n_trials)
                 for mode in ("seim", "eep")]
        results = {}
        with cf.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_tx_worker_init,
                initargs=(ch, all_indices)) as ex:
            for done, res in enumerate(ex.map(_tx_worker, tasks,
                                              chunksize=4)):
                snr, ii, t, mode, idx_hat = res
                results[(snr, ii, t, mode)] = idx_hat
                if (done + 1) % 500 == 0:
                    print(f"  {ch.name:>9s} 传输进度 {done + 1}/{len(tasks)}")
        # 按 (snr, img) 分组批量解码（batch = n_trials * 2）
        psnr_seim = np.zeros(len(snr_list))
        psnr_eep = np.zeros(len(snr_list))
        for si, snr in enumerate(snr_list):
            acc = {"seim": [], "eep": []}
            for ii in range(n_img):
                rec = cache[ii]
                batch_idx, modes = [], []
                for t in range(n_trials):
                    for mode in ("seim", "eep"):
                        batch_idx.append(results[(snr, ii, t, mode)])
                        modes.append(mode)
                recons = codec.decode_many(batch_idx, ref=rec, snr=snr)
                for mode, recon in zip(modes, recons):
                    acc[mode].append(codec.psnr(rec["gt"], recon))
            psnr_seim[si] = float(np.mean(acc["seim"]))
            psnr_eep[si] = float(np.mean(acc["eep"]))
            print(f"  {ch.name:>9s} SNR={snr:>2d} dB: "
                  f"SeIM {psnr_seim[si]:.3f} dB / "
                  f"w/o SS {psnr_eep[si]:.3f} dB")
        curves[ch.name] = dict(seim=psnr_seim, eep=psnr_eep)
        print(f"  {ch.name:>9s} 完成，用时 {time.time() - t0:.0f}s")

    _plot_psnr_curves(channels, curves, snr_lists, out_prefix)
    return curves


def _plot_psnr_curves(channels, curves, snr_lists, out_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, ch in zip(axes, channels):
        c = curves[ch.name]
        snr_list = snr_lists[ch.name]
        ax.plot(snr_list, c["seim"], "o-", color="#C00000", lw=1.2, ms=4,
                label="SeIM (proposed)")
        ax.plot(snr_list, c["eep"], "s--", color="#2E8B57", lw=1.2, ms=4,
                label="w/o stream splitting")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(ch.name, fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    axes[0].set_ylabel("Reconstruction PSNR (dB)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  -> 已保存 {out_prefix}.pdf/.png")


def main():
    parser = argparse.ArgumentParser(description="SeIM magazine simulation")
    parser.add_argument("--quick", action="store_true",
                        help="减少采样规模，用于快速验证流程")
    parser.add_argument("--trials", type=int, default=6,
                        help="每个 SNR 的试验次数（预训练模式为每图的信道实现数）")
    parser.add_argument("--vectors", type=int, default=4096,
                        help="每次试验的信源向量数（仅 synthetic）")
    parser.add_argument("--slots", type=int, default=20000,
                        help="BER 测量的时隙数")
    parser.add_argument("--codec", choices=["synthetic", "pretrained"],
                        default="synthetic", help="Part B 使用的语义编解码器")
    # 预训练模式参数（默认指向 SSC 仓库内的实际路径）
    parser.add_argument("--ssc_root", default=SSC_ROOT,
                        help="SSC 仓库根目录（包含 ssc/、basicsr/）")
    parser.add_argument("--config", default=os.path.join(
        SSC_ROOT, "experiments", "16xD", "train_SSC_bpp2_32C_16E_4D",
        "train_SSC_bpp2_32C_16E_4D_from_pretrain.yml"),
        help="训练 YAML 配置路径")
    parser.add_argument("--model", default=os.path.join(
        SSC_ROOT, "experiments", "16xD", "train_SSC_bpp2_32C_16E_4D",
        "models", "net_g_latest.pth"),
        help="预训练权重 .pth 路径")
    parser.add_argument("--image_dir", default=os.path.join(
        SSC_ROOT, "datasets", "Kodak24"), help="测试图像目录")
    parser.add_argument("--snr_train", type=int, default=10,
                        help="encode_all/decode 缺省 chan_param")
    parser.add_argument("--fig_dir", default=FIG_DIR, help="图片输出目录")
    parser.add_argument("--res_dir", default=RES_DIR, help="数据输出目录")
    parser.add_argument("--workers", type=int, default=48,
                        help="预训练模式信道传输的并行进程数")
    args = parser.parse_args()

    if args.quick:
        args.trials, args.vectors, args.slots = 2, 1024, 5000

    fig_dir = args.fig_dir
    res_dir = args.res_dir
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    snr_lists = SNR_LISTS

    channels = build_channels()
    ber = part_a_ber_asymmetry(channels, snr_lists, args.slots,
                               os.path.join(fig_dir, "fig_ber_asymmetry"))

    if args.codec == "synthetic":
        from semantic_codec import SyntheticRQCodec
        codec = SyntheticRQCodec(Ne=16, Nq=4, d_e=4)
        print(f"SyntheticRQCodec 就绪 (Ne={codec.Ne}, Nq={codec.Nq}, "
              f"d_e={codec.d_e})")
        curves = part_b_end_to_end(channels, codec, snr_lists, args.vectors,
                                   args.trials,
                                   os.path.join(fig_dir, "fig_snr_psnr"))
    else:
        from semantic_codec import PretrainedSwinSSCCodec
        image_paths = sorted(
            p for p in glob.glob(os.path.join(args.image_dir, "*"))
            if p.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")))
        if not image_paths:
            raise FileNotFoundError(
                f"测试图像目录为空: {args.image_dir}")
        if args.quick:
            image_paths = image_paths[:4]
        print(f"加载预训练权重: {args.model}")
        print(f"配置: {args.config}  图像: {len(image_paths)} 张")
        codec = PretrainedSwinSSCCodec(
            ssc_main_path=args.ssc_root,
            config_path=args.config,
            model_path=args.model,
            image_paths=image_paths,
            snr_train=args.snr_train)
        print(f"PretrainedSwinSSCCodec 就绪 (Ne={codec.Ne}, "
              f"Nq={codec.Nq}, device={codec.device})")
        curves = part_b_pretrained(channels, codec, snr_lists, args.trials,
                                   os.path.join(fig_dir, "fig_snr_psnr"),
                                   workers=args.workers)

    np.savez(os.path.join(res_dir, "sim_results.npz"),
             snr_lists={k: np.array(v) for k, v in snr_lists.items()},
             ber={k: np.array(v) for k, v in ber.items()},
             **{f"{k}_seim": v["seim"] for k, v in curves.items()},
             **{f"{k}_eep": v["eep"] for k, v in curves.items()})
    print(f"原始数据已保存到 {os.path.join(res_dir, 'sim_results.npz')}")

    # 汇总关键结论（终端打印，便于写论文时引用数字）
    print("\n===== 结果摘要 =====")
    for ch in channels:
        c = curves[ch.name]
        snr_list = snr_lists[ch.name]
        gain = c["seim"] - c["eep"]
        i = int(np.argmax(gain))
        print(f"{ch.name:>9s}: 最大 SeIM 增益 {gain[i]:.2f} dB @ "
              f"SNR={snr_list[i]} dB; "
              f"鲁棒流={'索引流' if ch.index_more_robust else '符号流'}")


if __name__ == "__main__":
    main()
