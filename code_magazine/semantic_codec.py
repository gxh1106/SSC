"""
semantic_codec.py
=================
语义编解码器接口：把"信源 <-> RQ 分层索引"抽象为统一 API，供
``run_simulation.py`` 与三种 IM 链路（im_channels.py）组合使用。

- ``SemanticCodecBase``     : 接口定义。
- ``SyntheticRQCodec``      : 默认实现（纯 NumPy）。合成高斯信源 +
                              Lloyd 算法逐层训练的残差量化（RQ）码本，
                              可直接复现 magazine 中的 SNR-PSNR 曲线。
- ``PretrainedSwinSSCCodec``: 预训练模型钩子（需要 PyTorch + SSC 仓库）。
                              加载 SSC 仓库 experiments/ 中训练好的
                              SwinSSC 权重，在真实图像（Kodak24）上评估
                              端到端 PSNR。用法见 README.md。

相对 SSC-magazine/sim/semantic_codec.py 的修正（已对照 SSC 仓库源码核实）：

1. [致命] 分辨率更新：原实现只在初始化时按训练尺寸（256x256）调用
   ``update_resolution``，而 Kodak 测试图（crop_divisor=128 裁剪后为
   512x768）与 attn_mask/input_resolution 不匹配，BasicBlock 内的
   ``assert L == H * W`` 会直接报错。现在 ``encode_all`` 按每张测试图
   的实际尺寸更新 encoder/decoder 分辨率。
2. [致命] 码本大小探测：原实现写
   ``quantizer.codebooks[0].embed.num_embeddings``，但 VQEmbedding 的
   ``embed`` 是一个方法（不是模块），会 AttributeError；且 VQEmbedding
   继承 ``nn.Embedding(n_embed + 1, ..., padding_idx=n_embed)``，
   即使取到 ``num_embeddings`` 也是 17 而非 16。现在直接读
   ``quantizer.n_embed[0]`` 与 ``quantizer.rq_depth``。
3. [行为] chan_param：原实现固定为 snr_train，而 SSC 官方推理
   （ssc/inference.py）是每个 SNR 都用 ``given_SNR=snr`` 驱动
   encoder/decoder 的 SNR 自适应模块（MVQ 模式的码本选择也依赖
   chan_param）。现在 ``encode_all`` / ``decode`` 均接受当前 SNR。
4. [健壮性] 兼容 encoder 返回 ``(x, mask)`` 的 SA/RA 变体；断言当前
   仅支持单头 RQ（num_heads=1 且非 MVQ 模式）；numpy 索引转 tensor 前
   强制 contiguous；接收端索引 clamp 到 [0, Ne) 防御越界。
"""

import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# 接口定义
# ---------------------------------------------------------------------------
class SemanticCodecBase:
    """语义编解码器统一接口。

    索引矩阵约定：形状 (L, Nq)，取值 [0, Ne)，第 q 列对应第 q 个 RQ
    量化步；列号越小语义越重要（由粗到精）。
    """

    Ne = 16   # 码本大小
    Nq = 4    # RQ 量化步数

    def generate_source(self, n, rng=None):
        """生成 n 个信源样本，返回信源张量。"""
        raise NotImplementedError

    def encode(self, source):
        """信源 -> (L, Nq) 整数索引矩阵。"""
        raise NotImplementedError

    def decode(self, indices, ref=None, snr=None):
        """（可能出错的）索引矩阵 -> 重建信源。
        ``ref`` 为 encode 时的原始信源（预训练实现需要它对齐 shape），
        ``snr`` 为当前信道 SNR（预训练实现用作 chan_param）。"""
        raise NotImplementedError

    def psnr(self, source, recon):
        """重建质量（dB），越大越好。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 默认实现：合成高斯信源 + Lloyd 训练的 RQ
# ---------------------------------------------------------------------------
class SyntheticRQCodec(SemanticCodecBase):
    """合成信源上的残差量化编解码器。

    信源 u ~ N(0, I_{d_e})；用 Lloyd 算法在训练集上逐层学习 RQ 码本
    （与 SSC 系统中 EMA 学习的码本作用一致）。PSNR 定义为
    10*log10(信号功率 / 每维均方误差)，单位方差信源下即 -10*log10(MSE)。
    """

    def __init__(self, Ne=16, Nq=4, d_e=4, train_size=60000,
                 kmeans_iters=30, seed=7):
        self.Ne, self.Nq, self.d_e = Ne, Nq, d_e
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((train_size, d_e))
        # 逐层 Lloyd 训练残差码本
        self.codebooks = []
        residual = X
        for _ in range(Nq):
            cb = self._lloyd(residual, Ne, kmeans_iters, rng)
            self.codebooks.append(cb)
            assign = self._nearest(residual, cb)
            residual = residual - cb[assign]
        self.codebooks = np.stack(self.codebooks)          # (Nq, Ne, d_e)

    @staticmethod
    def _nearest(X, cb):
        d = ((X[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    @classmethod
    def _lloyd(cls, X, K, iters, rng):
        cb = X[rng.choice(len(X), K, replace=False)].copy()
        for _ in range(iters):
            assign = cls._nearest(X, cb)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    cb[k] = X[mask].mean(0)
        return cb

    # -- 接口实现 ---------------------------------------------------------
    def generate_source(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return rng.standard_normal((n, self.d_e))

    def encode(self, source):
        X = np.asarray(source, dtype=np.float64)
        L = len(X)
        indices = np.empty((L, self.Nq), dtype=np.int64)
        residual = X.copy()
        for q in range(self.Nq):
            idx = self._nearest(residual, self.codebooks[q])
            indices[:, q] = idx
            residual -= self.codebooks[q][idx]
        return indices

    def decode(self, indices, ref=None, snr=None):
        indices = np.asarray(indices, dtype=np.int64)
        recon = np.zeros((len(indices), self.d_e))
        for q in range(self.Nq):
            recon += self.codebooks[q][indices[:, q]]
        return recon

    def psnr(self, source, recon):
        mse = float(np.mean((np.asarray(source) - np.asarray(recon)) ** 2))
        return 10.0 * np.log10(1.0 / max(mse, 1e-12))


# ---------------------------------------------------------------------------
# 预训练模型钩子：SSC 仓库的 SwinSSC + RQBottleneck（需要 PyTorch）
# ---------------------------------------------------------------------------
class PretrainedSwinSSCCodec(SemanticCodecBase):
    """加载 SSC 仓库中训练好的 SwinSSC 权重，在真实图像上评估。

    调用链与 SSC/ssc/inference.py、ssc/archs/SwinSSC_arch.py 的
    ``forward_faim`` eval 路径完全一致：

        encoder -> quantizer.ad -> embed_idxs (N, rq_depth)
        [numpy IM 信道传输, 见 im_channels.py]
        quantizer.embed(noisy_idxs) -> quantizer.da -> decoder -> recon

    参数
    ----
    ssc_main_path : str   SSC 仓库根目录（包含 ssc/、basicsr/）。
    config_path   : str   训练用的 YAML 配置（options/ 或 experiments/ 下）。
    model_path    : str   预训练权重 .pth（net_g_latest.pth 等）。
    image_paths   : list  测试图像路径列表（如 Kodak24）。
    snr_train     : int   默认 chan_param（不传 snr 时使用，默认 10）。
    device        : str   'cuda' 或 'cpu'。
    """

    def __init__(self, ssc_main_path, config_path, model_path,
                 image_paths, snr_train=10, device=None):
        for p in (ssc_main_path, os.path.join(ssc_main_path, 'ssc')):
            # ssc/ 子目录入路径：ssc/models/SSCGAN_model.py 内有
            # `from loss.distortion import Distortion` 的裸导入
            if p not in sys.path:
                sys.path.insert(0, p)
        import argparse

        import torch
        import yaml
        import torch.nn as nn
        from addict import Dict
        from ssc.archs.SwinSSC_arch import SwinSSC

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.chan_param = snr_train

        with open(config_path, 'r') as f:
            opt = Dict(yaml.safe_load(f))
        opt_args = opt.network_g.args
        image_dims = opt_args.image_dims
        C = int(opt_args.C)
        if opt_args.model_size == 'small':
            opt['network_g']['encoder_kwargs'] = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2],
                num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                patch_norm=True)
            opt['network_g']['decoder_kwargs'] = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2],
                num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                patch_norm=True)
        elif opt_args.model_size == 'base':
            opt['network_g']['encoder_kwargs'] = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                patch_norm=True)
            opt['network_g']['decoder_kwargs'] = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
                num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                patch_norm=True)
        elif opt_args.model_size == 'large':
            opt['network_g']['encoder_kwargs'] = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2],
                num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                patch_norm=True)
            opt['network_g']['decoder_kwargs'] = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2],
                num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                patch_norm=True)
        else:
            raise ValueError(f"未知 model_size: {opt_args.model_size}")

        net_cfg = opt['network_g']
        model = SwinSSC(args=argparse.Namespace(**net_cfg.get('args')),
                        encoder_kwargs=net_cfg.get('encoder_kwargs'),
                        decoder_kwargs=net_cfg.get('decoder_kwargs'),
                        rq_kwargs=net_cfg.get('rq_kwargs'))

        # 量化器模式检查：当前仅支持单头 RQ（非 MVQ）模式
        assert not model.quantizer.trainable_bit_flip_prob, \
            'MVQ (trainable_bit_flip_prob) 模式请改用 mvq_embed/mvq_da 路径'
        assert model.quantizer.num_heads == 1, \
            'MOC_RVQ (num_heads>1) 模式的多头索引重排请参见 quantizer.embed'

        # 与 inference.py 一致：先按训练尺寸初始化分辨率（attn_mask 随后被过滤）
        down = 2 ** model.downsample
        model.encoder.update_resolution(image_dims[1], image_dims[2])
        model.decoder.update_resolution(image_dims[1] // down,
                                        image_dims[2] // down)
        ckpt = torch.load(model_path,
                          map_location=lambda storage, loc: storage)
        key = 'params_ema' if 'params_ema' in ckpt else 'params'
        state = {k: v for k, v in ckpt[key].items() if 'attn_mask' not in k}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # 只剩 attn_mask 相关的 missing/unexpected 属正常
        real_missing = [k for k in missing if 'attn_mask' not in k]
        real_unexpected = [k for k in unexpected if 'attn_mask' not in k]
        if real_missing or real_unexpected:
            print(f'[警告] 权重加载存在非 attn_mask 的不匹配: '
                  f'missing={real_missing}, unexpected={real_unexpected}')
        self.model = model.eval().to(self.device)
        self.crop_divisor = opt['datasets'].get('val_1', {}).get(
            'crop_divisor', 128)

        # 码本参数（直接从 RQBottleneck 读取；VQEmbedding.embed 是方法，
        # 且 num_embeddings 含 padding 位为 n_embed+1，均不可用）
        self.Ne = int(self.model.quantizer.n_embed[0])
        self.Nq = int(self.model.quantizer.rq_depth)
        assert self.Ne == 2 ** int(round(np.log2(self.Ne))), \
            f'码本大小 Ne={self.Ne} 需为 2 的幂以匹配 IM 信道比特换算'

        # 加载测试图像
        self.images = [self._load_image(p) for p in image_paths]
        self._cache = {}
        self._cur_res = (image_dims[1], image_dims[2])
        # w/o_SAandRA 变体的 encoder/decoder 均不使用 chan_param（SNR 自
        # 适应模块被移除），RQ 模式的量化器亦不依赖 chan_param，因此编码
        # 结果与 SNR 无关，可跨 SNR 复用一次编码。
        self._snr_agnostic = (self.model.model == 'SwinJSCC_w/o_SAandRA')

    def _load_image(self, path):
        import cv2
        torch = self.torch
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        h, w, _ = img.shape
        th, tw = h - h % self.crop_divisor, w - w % self.crop_divisor
        top, left = (h - th) // 2, (w - tw) // 2
        img = img[top:top + th, left:left + tw, :]
        t = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1)))).float()
        return t.unsqueeze(0).to(self.device)

    def _ensure_resolution(self, H, W):
        """按当前输入尺寸更新 encoder/decoder 的 input_resolution 与
        attn_mask。SwinSSC.forward 在尺寸变化时也会做这件事；由于本类
        直接调用 encoder/decoder，必须自行保证尺寸一致。"""
        if (H, W) == self._cur_res:
            return H // (2 ** self.model.downsample), \
                   W // (2 ** self.model.downsample)
        Hf = H // (2 ** self.model.downsample)
        Wf = W // (2 ** self.model.downsample)
        self.model.encoder.update_resolution(H, W)
        self.model.decoder.update_resolution(Hf, Wf)
        self._cur_res = (H, W)
        return Hf, Wf

    # -- 接口实现 ---------------------------------------------------------
    def encode_all(self, snr=None):
        """对所有测试图像执行 encoder + RQ，缓存 (gt, indices, shape_info)。

        ``snr`` 为当前信道 SNR（对应官方推理的 given_SNR）；不传则使用
        初始化时的 snr_train。相同 (snr, 尺寸) 的结果会缓存复用。
        """
        torch = self.torch
        chan_param = self.chan_param if snr is None else snr
        key = 0 if self._snr_agnostic else chan_param
        if key in self._cache:
            return self._cache[key]
        cache = []
        with torch.no_grad():
            for img in self.images:
                B, _, H, W = img.shape
                Hf, Wf = self._ensure_resolution(H, W)
                feat = self.model.encoder(img, chan_param,
                                          self.model.channel_number,
                                          self.model.model)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]      # 兼容 w/_SA / w/_SAandRA 变体
                out = self.model.quantizer.ad(feat, feat_shape=(Hf, Wf),
                                              chan_param=chan_param)
                _, _, _, embed_idxs, shape_info = out
                cache.append(dict(
                    gt=img,
                    indices=np.ascontiguousarray(
                        embed_idxs.cpu().numpy().astype(np.int64)),
                    shape_info=shape_info,
                    chan_param=chan_param))
        self._cache[key] = cache
        return cache

    def generate_source(self, n, rng=None):
        raise NotImplementedError('预训练模式请使用 encode_all() 缓存的索引。')

    def encode(self, source):
        raise NotImplementedError('预训练模式请使用 encode_all() 缓存的索引。')

    def decode(self, indices, ref=None, snr=None):
        """ref 为 encode_all() 缓存中的单条记录（含 gt 与 shape_info）。
        ``snr`` 应与 encode_all 时使用的 chan_param 一致（默认取 ref 中
        记录的值）。"""
        torch = self.torch
        chan_param = snr if snr is not None else ref.get(
            'chan_param', self.chan_param)
        noisy = torch.from_numpy(np.ascontiguousarray(
            np.asarray(indices, dtype=np.int64))).to(self.device)
        noisy = noisy.clamp_(0, self.Ne - 1)    # 防御：索引越界保护
        gt = ref['gt']
        B, _, H, W = gt.shape
        self._ensure_resolution(H, W)           # 解码器分辨率同样需对齐
        with torch.no_grad():
            quant = self.model.quantizer.embed(noisy,
                                               chan_param=chan_param)
            feat_dq = self.model.quantizer.da(quant, ref['shape_info'])
            recon = self.model.decoder(feat_dq, chan_param,
                                       self.model.model)
        if isinstance(recon, (tuple, list)):
            recon = recon[0]
        return torch.clamp(recon, 0, 1)

    def decode_many(self, indices_list, ref, snr=None):
        """批量解码：把多条 (L, Nq) 索引沿 N 维拼接成 batch 一次性过
        decoder，显著减少 GPU 调用次数。返回 recon tensor 列表。"""
        torch = self.torch
        chan_param = snr if snr is not None else ref.get(
            'chan_param', self.chan_param)
        arrs = [np.ascontiguousarray(np.asarray(x, dtype=np.int64))
                for x in indices_list]
        B = len(arrs)
        noisy = torch.from_numpy(np.concatenate(arrs, axis=0)).to(self.device)
        noisy = noisy.clamp_(0, self.Ne - 1)
        gt = ref['gt']
        _, _, H, W = gt.shape
        self._ensure_resolution(H, W)
        si = ref['shape_info']                      # (1, L, C)
        shape_info = (B, si[1], si[2])
        with torch.no_grad():
            quant = self.model.quantizer.embed(noisy,
                                               chan_param=chan_param)
            feat_dq = self.model.quantizer.da(quant, shape_info)
            recon = self.model.decoder(feat_dq, chan_param,
                                       self.model.model)
        if isinstance(recon, (tuple, list)):
            recon = recon[0]
        recon = torch.clamp(recon, 0, 1)
        return [recon[i:i + 1] for i in range(B)]

    def psnr(self, source, recon):
        torch = self.torch
        mse = torch.mean((source.to(torch.float64)
                          - recon.to(torch.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return float(20 * torch.log10(1.0 / torch.sqrt(mse)))
