"""
run_config_sweep.py
===================
索引/符号比特搭配（m1 vs m2）扫描实验。

在保持各 IM 家族物理模型不变的前提下，改变每资源单元的索引比特数 m1
（激活模式数）与符号比特数 m2（星座阶数），考察：
  1) 双流 BER 不对称的方向与强度如何随配置变化；
  2) 端到端 SeIM 增益如何随之变化（验证"gain 跟踪不对称强度"）。

每个配置：离线 profiling 判定鲁棒流（2e4 时隙/点），然后在 Kodak24 +
预训练 SwinSSC 上测 SeIM vs w/o SS 的 PSNR（trials=3，配对比较）。

用法
----
    python run_config_sweep.py --workers 48
"""

import argparse
import concurrent.futures as cf
import os
import time

import numpy as np

np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from im_channels import FAIMChannel, SMChannel, OFDMIMChannel, decide_robust_stream
from semantic_codec import PretrainedSwinSSCCodec

HERE = os.path.dirname(os.path.abspath(__file__))
SSC_ROOT = os.path.normpath(os.path.join(HERE, ".."))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 120,
})

# ---------------------------------------------------------------------------
# 扫描配置：每个家族在可比总速率下调整 (m1, m2) 搭配
# ---------------------------------------------------------------------------
CONFIGS = [
    # (标签, 信道构造 callable, 评估 SNR 点)
    ("SM 2+6",  lambda s: SMChannel(Nt=4, Nr=4, M=64, seed=s),
     list(range(2, 26, 4))),
    ("SM 4+4",  lambda s: SMChannel(Nt=16, Nr=4, M=16, seed=s),
     list(range(2, 26, 4))),
    ("SM 2+4",  lambda s: SMChannel(Nt=4, Nr=4, M=16, seed=s),
     list(range(2, 26, 4))),
    ("FA-IM 1+6", lambda s: FAIMChannel(Ns=2, Np=16, Nr=8, M=64, W=2.0,
                                        L_paths=10, seed=s),
     list(range(0, 22, 4))),
    ("FA-IM 2+6", lambda s: FAIMChannel(Ns=4, Np=16, Nr=8, M=64, W=2.0,
                                        L_paths=10, seed=s),
     list(range(0, 22, 4))),
    ("FA-IM 3+4", lambda s: FAIMChannel(Ns=8, Np=16, Nr=8, M=16, W=2.0,
                                        L_paths=10, seed=s),
     list(range(0, 22, 4))),
    ("OFDM-IM 2+4", lambda s: OFDMIMChannel(n=4, k=2, M=4, seed=s),
     list(range(10, 32, 4))),
    ("OFDM-IM 2+8", lambda s: OFDMIMChannel(n=4, k=2, M=16, seed=s),
     list(range(10, 32, 4))),
]

FAMILY_COLORS = {"SM": "#1f77b4", "FA-IM": "#9467bd", "OFDM-IM": "#8c564b"}

# ---------------------------------------------------------------------------
# 并行传输 worker（与 run_simulation.py 相同模式）
# ---------------------------------------------------------------------------
_WCH = None
_WIDX = None


def _tx_worker_init(ch, all_indices):
    global _WCH, _WIDX
    _WCH, _WIDX = ch, all_indices


def _tx_worker(task):
    snr, ii, trial, mode, seed = task
    rng = np.random.default_rng(seed)
    return (snr, ii, trial, mode, _WCH.transmit(_WIDX[ii], snr,
                                                mode=mode, rng=rng))


def eval_config(label, ctor, snr_list, cache, codec, n_trials, workers,
                profile_slots):
    """单个配置：profiling + 端到端 PSNR。返回 dict。"""
    ch = ctor(0)
    robust, ber_i, ber_s = decide_robust_stream(ch, snr_list, profile_slots)
    ch.index_more_robust = robust

    all_indices = [rec["indices"] for rec in cache]
    n_img = len(cache)
    tasks = [(snr, ii, t, mode,
              (si * 100000 + ii * 100 + t * 10) & 0x7fffffff)
             for si, snr in enumerate(snr_list)
             for ii in range(n_img)
             for t in range(n_trials)
             for mode in ("seim", "eep")]
    results = {}
    with cf.ProcessPoolExecutor(max_workers=workers,
                                initializer=_tx_worker_init,
                                initargs=(ch, all_indices)) as ex:
        for res in ex.map(_tx_worker, tasks, chunksize=4):
            snr, ii, t, mode, idx_hat = res
            results[(snr, ii, t, mode)] = idx_hat

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
    gain = psnr_seim - psnr_eep
    i = int(np.argmax(gain))
    print(f"  {label:>12s}: 鲁棒流={'索引' if robust else '符号'}流, "
          f"最大增益 {gain[i]:.2f} dB @ {snr_list[i]} dB")
    return dict(label=label, robust=robust, snr=np.array(snr_list),
                ber_i=np.array(ber_i), ber_s=np.array(ber_s),
                seim=psnr_seim, eep=psnr_eep, gain=gain)


def main():
    parser = argparse.ArgumentParser(description="SeIM m1/m2 config sweep")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile_slots", type=int, default=20000)
    parser.add_argument("--ssc_root", default=SSC_ROOT)
    parser.add_argument("--config", default=os.path.join(
        SSC_ROOT, "experiments", "16xD", "train_SSC_bpp2_32C_16E_4D",
        "train_SSC_bpp2_32C_16E_4D_from_pretrain.yml"))
    parser.add_argument("--model", default=os.path.join(
        SSC_ROOT, "experiments", "16xD", "train_SSC_bpp2_32C_16E_4D",
        "models", "net_g_latest.pth"))
    parser.add_argument("--image_dir", default=os.path.join(
        SSC_ROOT, "datasets", "Kodak24"))
    parser.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()

    import glob
    image_paths = sorted(p for p in glob.glob(os.path.join(args.image_dir, "*"))
                         if p.lower().endswith((".png", ".jpg", ".jpeg")))
    codec = PretrainedSwinSSCCodec(ssc_main_path=args.ssc_root,
                                   config_path=args.config,
                                   model_path=args.model,
                                   image_paths=image_paths, snr_train=10)
    cache = codec.encode_all(snr=10)   # SNR 无关，仅编码一次
    print(f"编码完成：{len(cache)} 张图像")

    all_results = []
    for label, ctor, snr_list in CONFIGS:
        t0 = time.time()
        r = eval_config(label, ctor, snr_list, cache, codec, args.trials,
                        args.workers, args.profile_slots)
        all_results.append(r)
        print(f"           用时 {time.time() - t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "config_sweep.npz"),
             results=np.array(all_results, dtype=object))

    # 绘图：每个家族一个子图，gain vs SNR，每条曲线一个配置
    families = ["SM", "FA-IM", "OFDM-IM"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    styles = ["o-", "s--", "^-."]
    for ax, fam in zip(axes, families):
        j = 0
        for r in all_results:
            if not r["label"].startswith(fam):
                continue
            ax.plot(r["snr"], r["gain"], styles[j % 3], lw=1.2, ms=4,
                    label=f'{r["label"].split()[-1]} bits')
            j += 1
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(fam, fontsize=9)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=7, title="(index+symbol)", title_fontsize=7)
    axes[0].set_ylabel("SeIM gain (dB)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"fig_config_sweep.{ext}"),
                    dpi=300)
    print(f"已保存 {os.path.join(args.out_dir, 'fig_config_sweep.pdf/.png')}")


if __name__ == "__main__":
    main()
