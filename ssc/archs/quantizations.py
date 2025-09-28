# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

def decimal_to_binary_rows(decimal_tensor, num_bits):
    """
    将一维的十进制整数张量转换为二维的二进制张量。
    """
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=decimal_tensor.device, dtype=decimal_tensor.dtype)
    return decimal_tensor.unsqueeze(-1).bitwise_and(mask).ne(0).int()

def binary_rows_to_decimal(binary_tensor):
    """
    将二维的二进制张量转换为一维的十进制整数张量。
    """
    num_bits = binary_tensor.shape[1]
    powers_of_two = 2 ** torch.flip(torch.arange(num_bits, dtype=binary_tensor.dtype, device=binary_tensor.device), dims=(0,))
    decimal_values = torch.sum(binary_tensor * powers_of_two, dim=1)
    return decimal_values.long()

def calculate_p_from_snr_db(snr_db):
    """
    根据给定的SNR（单位：dB），计算BPSK在AWGN信道下的比特错误概率p。

    参数:
        snr_db (torch.Tensor or float): 一个或多个信噪比的值，以dB为单位。

    返回:
        torch.Tensor: 对应的比特错误概率 p。
    """
    # 确保输入是torch.Tensor
    if not isinstance(snr_db, torch.Tensor):
        snr_db = torch.tensor(snr_db, dtype=torch.float32)

    # 1. 将SNR从dB转换为线性值 (SNR_linear = 10^(SNR_dB / 10))
    snr_linear = 10.0 ** (snr_db / 10.0)

    # 2. 对于BPSK调制，Eb/N0 在数值上等于 SNR_linear
    eb_n0 = snr_linear

    # 3. 使用公式 p = 0.5 * erfc(sqrt(Eb/N0)) 计算比特错误概率
    # torch.erfc 是PyTorch中的互补误差函数
    p = 0.5 * torch.erfc(torch.sqrt(eb_n0))
    
    return p

class BSC_channel(nn.Module):
    def __init__(self, bit_flip_prob=0) -> None:
        super().__init__()
        self.bit_flip_prob = bit_flip_prob

    def forward(self, x, bit_flip_prob=None):  # this func is compatible with {0, 1} bits, caution about the input format

        if bit_flip_prob is not None:
            self.bit_flip_prob = bit_flip_prob
        
        out = x.clone()
        noise = torch.rand_like(x.float()) < self.bit_flip_prob
        out[noise] = 1 - out[noise]

        return dict(
            out=out,
            bit_flip_prob=self.bit_flip_prob
        )
    

class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]
        return distances

    # @torch.no_grad()
    # def find_nearest_embedding(self, inputs):
    #     distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
    #     embed_idxs = distances.argmin(dim=-1)  # use padding index or not

    #     return embed_idxs
    @torch.no_grad()
    def find_nearest_embedding(self, inputs, chunk_size=256):
        """
        Finds the nearest embedding indices without creating the full distance matrix.
        """
        # 准备工作
        codebook_t = self.weight[:-1, :].t()
        (embed_dim, n_embed) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)
        n_vectors = inputs_flat.shape[0]

        # 初始化用于追踪最小距离和对应索引的张量
        # 用无穷大来初始化最小距离
        min_distances = torch.full((n_vectors,), float('inf'), device=inputs.device, dtype=inputs.dtype)
        embed_idxs = torch.zeros(n_vectors, device=inputs.device, dtype=torch.long)
        
        # 预先计算输入向量的范数平方
        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)

        # 迭代计算
        for i in range(0, n_embed, chunk_size):
            # 获取当前码本块
            codebook_t_chunk = codebook_t[:, i:i+chunk_size]
            
            # 计算当前块的距离
            codebook_t_norm_sq_chunk = codebook_t_chunk.pow(2.).sum(dim=0, keepdim=True)
            distances_chunk = torch.addmm(
                inputs_norm_sq + codebook_t_norm_sq_chunk,
                inputs_flat,
                codebook_t_chunk,
                alpha=-2.0,
            )
            
            # 在当前块内寻找最小距离及其相对索引
            chunk_min_distances, chunk_embed_idxs = distances_chunk.min(dim=1)
            
            # 确定哪些向量在当前块中找到了更近的码字
            update_mask = chunk_min_distances < min_distances
            
            # 更新全局最小距离
            min_distances[update_mask] = chunk_min_distances[update_mask]
            
            # 更新全局索引，注意要加上块的偏移量 i
            embed_idxs[update_mask] = chunk_embed_idxs[update_mask] + i
                
        # 将最终的索引 reshape 成输入的空间形状
        embed_idxs = embed_idxs.reshape(*inputs_shape[:-1])
        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x    
    
    # @torch.no_grad()
    # def _update_buffers(self, vectors, idxs):

    #     n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]
        
    #     vectors = vectors.reshape(-1, embed_dim)
    #     idxs = idxs.reshape(-1)
        
    #     n_vectors = vectors.shape[0]
    #     n_total_embed = n_embed

    #     one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
    #     one_hot_idxs.scatter_(dim=0,
    #                           index=idxs.unsqueeze(0),
    #                           src=vectors.new_ones(1, n_vectors)
    #                           )

    #     cluster_size = one_hot_idxs.sum(dim=1)
    #     vectors_sum_per_cluster = one_hot_idxs @ vectors

    #     if dist.is_initialized():
    #         dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
    #         dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

    #     self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
    #     self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        
    #     if self.restart_unused_codes:
    #         if n_vectors < n_embed:
    #             vectors = self._tile_with_noise(vectors, n_embed)
    #         n_vectors = vectors.shape[0]
    #         _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            
    #         if dist.is_initialized():
    #             dist.broadcast(_vectors_random, 0)
        
    #         usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
    #         self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
    #         self.cluster_size_ema.mul_(usage.view(-1))
    #         self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))
    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]
        
        vectors_flat = vectors.reshape(-1, embed_dim)
        idxs_flat = idxs.reshape(-1)
        
        n_vectors = vectors_flat.shape[0]
        
        # --- 高效的、避免创建巨大矩阵的实现方式 ---
        # 1. 计算 cluster_size
        # torch.bincount 会统计每个索引出现的次数
        cluster_size = torch.bincount(idxs_flat, minlength=n_embed).float()
        
        # 2. 计算 vectors_sum_per_cluster
        # 创建一个零矩阵用于接收向量和
        vectors_sum_per_cluster = vectors_flat.new_zeros(n_embed, embed_dim)
        # 使用 scatter_add_ 将向量加到其对应索引的位置
        # 这等效于 one_hot_idxs @ vectors，但完全不创建 one_hot_idxs 矩阵
        vectors_sum_per_cluster.scatter_add_(dim=0, 
                                            index=idxs_flat.unsqueeze(1).expand_as(vectors_flat), 
                                            src=vectors_flat)
        # --- 实现结束 ---

        # 后续的分布式训练和 EMA 更新逻辑保持不变
        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        
        # 重启未使用码本的逻辑保持不变
        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors_flat = self._tile_with_noise(vectors_flat, n_embed)
            n_vectors = vectors_flat.shape[0]
            _vectors_random = vectors_flat[torch.randperm(n_vectors, device=vectors_flat.device)][:n_embed]
            
            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
        
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds

class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, C)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 latent_shape,
                 embed_dim,
                 n_embed,
                 rq_depth,
                 num_quant=1,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 unembed_dim=None
                 ):
        super().__init__()

        self.latent_shape = torch.Size(latent_shape)
        self.embed_dim = embed_dim
        self.rq_depth = rq_depth

        if unembed_dim is not None:
            self.unembed_dim = unembed_dim
        else:
            self.unembed_dim = num_quant * self.latent_shape[-1]

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.rq_depth)]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.rq_depth)]
        assert len(self.n_embed) == self.rq_depth
        assert len(self.decay) == self.rq_depth

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0], 
                                    self.embed_dim, 
                                    decay=self.decay[0], 
                                    restart_unused_codes=restart_unused_codes,
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.rq_depth)])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx], 
                                     self.embed_dim, 
                                     decay=self.decay[idx], 
                                     restart_unused_codes=restart_unused_codes,
                                     ) for idx in range(self.rq_depth)]
            self.codebooks = nn.ModuleList(codebooks)

        # 作用于 (B, C, L) 形状的张量
        self.pre_quant = nn.Conv1d(
            in_channels=self.latent_shape[-1],
            out_channels=self.unembed_dim * self.embed_dim,
            kernel_size=1,
            # groups=self.latent_shape[-1]
        )
        self.post_quant = nn.Conv1d(
            in_channels=self.unembed_dim * self.embed_dim,
            out_channels=self.latent_shape[-1],
            kernel_size=1,
            # groups=self.latent_shape[-1]
        )

        # self.channel = BSC_channel(stochastic=False, bit_flip_prob=3.9e-6)
        self.channel = BSC_channel(bit_flip_prob=3.9e-6)
        self.num_bits_per_level = [(n_embed_i - 1).bit_length() for n_embed_i in self.n_embed[:self.rq_depth]]

    def to_code_shape(self, x):
        # 输入形状为 (B, L, C)
        x = x.permute(0, 2, 1).contiguous()  # 形状变为 (B, C, L)
        # (B, C, L) -> (B, num_quant * embed_dim * C, L)
        x = self.pre_quant(x)
        # (B, num_quant * embed_dim * C, L) -> (B, L, num_quant * embed_dim * C)
        x = x.permute(0, 2, 1).contiguous()
        # (B, L, num_quant * embed_dim * C) -> (B * L * num_quant * C, embed_dim)
        x_flatten = x.reshape(-1, self.embed_dim)

        return x_flatten

    def to_latent_shape(self, x, B, L):
        # x 是量化器的输出, 形状为 (B * L * num_quant * C, embed_dim)
        assert x.shape[0] == B * L * self.unembed_dim, f"Input sequence length {x.shape[0]} does not match B*L*C ({B*L*self.unembed_dim})"
        # (B * L * num_quant * C, embed_dim) -> (B, L, embed_dim * num_quant * C)
        x = x.view(B, L, self.embed_dim * self.unembed_dim)
        # (B, L, num_quant * embed_dim * C) -> (B, num_quant * embed_dim * C, L)
        x = x.permute(0, 2, 1).contiguous()
        # (B, num_quant * embed_dim * C, L) -> (B, C, L)
        x = self.post_quant(x)
        # (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1).contiguous()

        return x

    def quantize(self, x):
        r"""
        Performs residual quantization on a 2D batch of vectors.
        Shape:
            - x: (N, embed_dim)
            - quant_list[i]: (N, embed_dim)
            - embed_idxs: (N, d)
        """
        residual_feature = x.detach().clone()

        quant_list = []
        embed_idxs_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.rq_depth):
            quant, embed_idx = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            embed_idxs_list.append(embed_idx.unsqueeze(-1))
        
        embed_idxs = torch.cat(embed_idxs_list, dim=-1)
        return quant_list, embed_idxs

    def forward(self, x):
        B, L, C = x.shape
        x_reshaped = self.to_code_shape(x)
        quant_list, embed_idxs = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        quants_trunc = self.to_latent_shape(quant_list[-1], B, L)
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, embed_idxs

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list1 = []
        for _, quant in enumerate(quant_list):
            partial_loss1 = (x-quant.detach()).pow(2.0).mean()
            loss_list1.append(partial_loss1)
            # partial_loss2 = (x.detach()-quant).pow(2.0).mean()
            # loss_list2.append(partial_loss2)

        # commitment_loss = torch.mean(torch.stack(loss_list1)) + 0.25 * torch.mean(torch.stack(loss_list2))
        commitment_loss = torch.mean(torch.stack(loss_list1))
        return commitment_loss
    
    def ad(self, x, feat_shape=None):
        shape_info = x.shape
        x_reshaped = self.to_code_shape(x)
        quant_list, embed_idxs = self.quantize(x_reshaped)
        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)

        return x_reshaped, quant_list[-1], commitment_loss, embed_idxs, shape_info
    
    def da(self, quant_recon, shape_info=None):

        feature_dequant = self.to_latent_shape(quant_recon, shape_info[0], shape_info[1])

        return feature_dequant
    
    def feature_pass_channel(self, embed_idxs, chan_param, noise_config=None):
        """
        将嵌入索引转换为比特流，通过BSC信道，再转换回索引，并重构量化矢量。
        可以根据 noise_config 对特定层施加更强的噪声。

        参数:
            embed_idxs (torch.Tensor): 形状为 [N, rq_depth] 的码本索引。
            chan_param (float): 用于计算基础比特错误率的信道参数 (例如 SNR in dB)。
            noise_config (dict, optional): 指定噪声注入策略的字典。
                例如: {'target_layer': 0, 'noise_factor': 10}
                - 'target_layer' (int): 要施加更强噪声的层级索引。
                - 'noise_factor' (float): 噪声增强因子，p_new = p_base * noise_factor。
                默认为 None，表示所有层使用相同的噪声水平。

        返回:
            torch.Tensor: 重构后的量化矢量。
        """
        # 1. 根据SNR计算基础的比特错误概率
        p_base = calculate_p_from_snr_db(chan_param)

        noisy_idxs_list = []
        
        # 2. 逐层处理：转换比特 -> 施加噪声 -> 转换回索引
        for i in range(self.rq_depth):
            # --- a. 将当前层的索引转换为比特流 ---
            indices_level_i = embed_idxs[:, i]
            num_bits = self.num_bits_per_level[i]
            binary_tensor = decimal_to_binary_rows(indices_level_i, num_bits)

            # --- b. 确定当前层的噪声水平 ---
            p_final = p_base # 默认使用基础噪声
            
            # 如果提供了噪声配置，并且当前层是目标层
            if noise_config and noise_config.get('target_layer') == i:
                noise_factor = noise_config.get('noise_factor', 1.0)
                p_final = p_base * noise_factor
                # 钳位操作：比特错误率p不应超过0.5
                p_final = torch.clamp(p_final, max=0.5) 
                # print(f"Layer {i}: Applying stronger noise. Base p: {p_base.item():.2e}, Final p: {p_final.item():.2e}")

            # --- c. 将当前层的比特流独立地通过BSC信道 ---
            channel_output = self.channel(binary_tensor, bit_flip_prob=p_final)
            noisy_bits = channel_output['out']

            # --- d. 将带噪比特转换回整数索引 ---
            n_embed_i = self.n_embed[i]
            noisy_indices_level_i = binary_rows_to_decimal(noisy_bits)
            
            # 钳位操作，防止因比特错误导致的索引越界
            noisy_indices_level_i = torch.clamp(noisy_indices_level_i, 0, n_embed_i - 1)
            
            noisy_idxs_list.append(noisy_indices_level_i.unsqueeze(1))

        # 3. 将各层得到的带噪索引拼接起来
        noisy_idxs = torch.cat(noisy_idxs_list, dim=1)

        # 4. 从新的（带噪）索引重构量化矢量
        N, _ = embed_idxs.shape
        quant_recon = torch.zeros(N, self.embed_dim, device=embed_idxs.device)
        for i in range(self.rq_depth):
            embeds = self.codebooks[i].embed(noisy_idxs[:, i])
            quant_recon.add_(embeds)

        return quant_recon

    def embed(self, noisy_idxs):
        # 4. 从新的（带噪）索引重构量化矢量
        N, _ = noisy_idxs.shape
        quant_recon = torch.zeros(N, self.embed_dim, device=noisy_idxs.device)
        for i in range(self.rq_depth):
            embeds = self.codebooks[i].embed(noisy_idxs[:, i])
            quant_recon.add_(embeds)

        return quant_recon
    

class RQBottleneck_ConQuant(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, C)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 latent_shape,
                 embed_dim,
                 n_embed,
                 rq_depth,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True
                 ):
        super().__init__()

        self.latent_shape = torch.Size(latent_shape)
        self.embed_dim = embed_dim
        self.rq_depth = rq_depth

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.rq_depth)]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.rq_depth)]
        assert len(self.n_embed) == self.rq_depth
        assert len(self.decay) == self.rq_depth

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0], 
                                    self.embed_dim, 
                                    decay=self.decay[0], 
                                    restart_unused_codes=restart_unused_codes,
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.rq_depth)])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx], 
                                     self.embed_dim, 
                                     decay=self.decay[idx], 
                                     restart_unused_codes=restart_unused_codes,
                                     ) for idx in range(self.rq_depth)]
            self.codebooks = nn.ModuleList(codebooks)

        self.pre_quant = nn.Conv2d(
            self.latent_shape[-1],
            self.embed_dim * self.latent_shape[-1],
            kernel_size=1
        )
        self.post_quant = nn.Conv2d(
            self.embed_dim * self.latent_shape[-1],
            self.latent_shape[-1],
            kernel_size=1
        )

        # self.channel = BSC_channel(stochastic=False, bit_flip_prob=3.9e-6)
        self.channel = BSC_channel(bit_flip_prob=3.9e-6)
        self.num_bits_per_level = [(n_embed_i - 1).bit_length() for n_embed_i in self.n_embed[:self.rq_depth]]

    def to_code_shape(self, x):
        if x.ndim == 3:
            # 输入是 (B, L, C)，其中 L = H * W
            B, L, C = x.shape
            H, W = self.latent_shape[0], self.latent_shape[1]
            assert L == H * W, f"Input sequence length {L} does not match H*W ({H*W})"
            x = x.permute(0, 2, 1).view(B, C, H, W).contiguous()

        x = self.pre_quant(x)  # transform shape from [batch_size, channel, height, width] to [batch_size, embed_dim * channel, height, width]

        x = torch.permute(x, (0, 2, 3, 1))  # shape [batch_size, height, width, embed_dim * channel]
        x_flatten = torch.reshape(x, (-1, self.embed_dim))  # shape [batch_size * height * width * embed_dim * channel, embed_dim]

        return x_flatten

    def to_latent_shape(self, x):
        L = x.shape[0]
        (H, W, C) = self.latent_shape
        B = L // (H * W * C)
        # (B, H, W, embed_dim * C) -> (B, embed_dim * C, H, W)
        x = x.view(B, H, W, self.embed_dim * C).permute(0, 3, 1, 2).contiguous()

        # (B, C, H, W)
        x = self.post_quant(x)

        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # (B, H*W, C)
        x = x.view(B, H * W, C)

        return x

    def quantize(self, x):
        r"""
        Performs residual quantization on a 2D batch of vectors.
        Shape:
            - x: (N, embed_dim)
            - quant_list[i]: (N, embed_dim)
            - embed_idxs: (N, d)
        """
        residual_feature = x.detach().clone()

        quant_list = []
        embed_idxs_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.rq_depth):
            quant, embed_idx = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            embed_idxs_list.append(embed_idx.unsqueeze(-1))
        
        embed_idxs = torch.cat(embed_idxs_list, dim=-1)
        return quant_list, embed_idxs

    def forward(self, x):
        x_reshaped = self.to_code_shape(x)
        quant_list, embed_idxs = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        quants_trunc = self.to_latent_shape(quant_list[-1])
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, embed_idxs

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list1 = []
        loss_list2 = []
        for _, quant in enumerate(quant_list):
            partial_loss1 = (x-quant.detach()).pow(2.0).mean()
            loss_list1.append(partial_loss1)
            # partial_loss2 = (x.detach()-quant).pow(2.0).mean()
            # loss_list2.append(partial_loss2)

        # commitment_loss = torch.mean(torch.stack(loss_list1)) + 0.25 * torch.mean(torch.stack(loss_list2))
        commitment_loss = torch.mean(torch.stack(loss_list1))
        return commitment_loss
    
    def ad(self, x, feat_shape=None):
        if self.training:
            # --- 训练路径 ---
            x_reshaped = self.to_code_shape(x)
            shape_info = None
        else:
            # --- 验证/评估路径 ---
            if feat_shape is None:
                raise ValueError("`feat_shape` (H_feat, W_feat) must be provided during evaluation.")
            x_reshaped, shape_info = self._encode_dynamic_3d(x, feat_shape)

        quant_list, embed_idxs = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)

        return x_reshaped, quant_list[-1], commitment_loss, embed_idxs, shape_info
    
    def da(self, quant_recon, shape_info=None):
        if shape_info is None:
            # --- 训练路径 ---
            assert self.training, "shape_info is required during evaluation mode."
            feature_dequant = self.to_latent_shape(quant_recon)
        else:
            # --- 验证/评估路径 ---
            feature_dequant = self._decode_dynamic_3d(quant_recon, shape_info)
        return feature_dequant
    
    def feature_pass_channel(self, embed_idxs, chan_param, noise_config=None):
        """
        将嵌入索引转换为比特流，通过BSC信道，再转换回索引，并重构量化矢量。
        可以根据 noise_config 对特定层施加更强的噪声。

        参数:
            embed_idxs (torch.Tensor): 形状为 [N, rq_depth] 的码本索引。
            chan_param (float): 用于计算基础比特错误率的信道参数 (例如 SNR in dB)。
            noise_config (dict, optional): 指定噪声注入策略的字典。
                例如: {'target_layer': 0, 'noise_factor': 10}
                - 'target_layer' (int): 要施加更强噪声的层级索引。
                - 'noise_factor' (float): 噪声增强因子，p_new = p_base * noise_factor。
                默认为 None，表示所有层使用相同的噪声水平。

        返回:
            torch.Tensor: 重构后的量化矢量。
        """
        # 1. 根据SNR计算基础的比特错误概率
        p_base = calculate_p_from_snr_db(chan_param)

        noisy_idxs_list = []
        
        # 2. 逐层处理：转换比特 -> 施加噪声 -> 转换回索引
        for i in range(self.rq_depth):
            # --- a. 将当前层的索引转换为比特流 ---
            indices_level_i = embed_idxs[:, i]
            num_bits = self.num_bits_per_level[i]
            binary_tensor = decimal_to_binary_rows(indices_level_i, num_bits)

            # --- b. 确定当前层的噪声水平 ---
            p_final = p_base # 默认使用基础噪声
            
            # 如果提供了噪声配置，并且当前层是目标层
            if noise_config and noise_config.get('target_layer') == i:
                noise_factor = noise_config.get('noise_factor', 1.0)
                p_final = p_base * noise_factor
                # 钳位操作：比特错误率p不应超过0.5
                p_final = torch.clamp(p_final, max=0.5) 
                # print(f"Layer {i}: Applying stronger noise. Base p: {p_base.item():.2e}, Final p: {p_final.item():.2e}")

            # --- c. 将当前层的比特流独立地通过BSC信道 ---
            channel_output = self.channel(binary_tensor, bit_flip_prob=p_final)
            noisy_bits = channel_output['out']

            # --- d. 将带噪比特转换回整数索引 ---
            n_embed_i = self.n_embed[i]
            noisy_indices_level_i = binary_rows_to_decimal(noisy_bits)
            
            # 钳位操作，防止因比特错误导致的索引越界
            noisy_indices_level_i = torch.clamp(noisy_indices_level_i, 0, n_embed_i - 1)
            
            noisy_idxs_list.append(noisy_indices_level_i.unsqueeze(1))

        # 3. 将各层得到的带噪索引拼接起来
        noisy_idxs = torch.cat(noisy_idxs_list, dim=1)

        # 4. 从新的（带噪）索引重构量化矢量
        N, _ = embed_idxs.shape
        quant_recon = torch.zeros(N, self.embed_dim, device=embed_idxs.device)
        for i in range(self.rq_depth):
            embeds = self.codebooks[i].embed(noisy_idxs[:, i])
            quant_recon.add_(embeds)

        return quant_recon
    # def feature_pass_channel(self, embed_idxs, chan_param):
    #     p_value = calculate_p_from_snr_db(chan_param)
    #     all_bit_streams = []
    #     # 将整数索引转换为比特流
    #     for i in range(self.rq_depth):
    #         indices_level_i = embed_idxs[:, i]
    #         num_bits = self.num_bits_per_level[i]

    #         binary_tensor = decimal_to_binary_rows(indices_level_i, num_bits)
    #         all_bit_streams.append(binary_tensor)
    #     concatenated_bits = torch.cat(all_bit_streams, dim=1)

    #     # 将比特流通过BSC信道
    #     channel_output = self.channel(concatenated_bits, bit_flip_prob=p_value)
    #     noisy_bits = channel_output['out']

    #     # 将带噪比特转换回整数索引
    #     noisy_idxs_list = []
    #     current_pos = 0
    #     for i in range(self.rq_depth):
    #         num_bits = self.num_bits_per_level[i]
    #         n_embed_i = self.n_embed[i]
            
    #         bit_segment = noisy_bits[:, current_pos : current_pos + num_bits]
    #         current_pos += num_bits

    #         noisy_indices_level_i = binary_rows_to_decimal(bit_segment)
    #         # 钳位操作，防止因比特错误导致的索引越界
    #         noisy_indices_level_i = torch.clamp(noisy_indices_level_i, 0, n_embed_i - 1)
            
    #         noisy_idxs_list.append(noisy_indices_level_i.unsqueeze(1))

    #     noisy_idxs = torch.cat(noisy_idxs_list, dim=1)

    #     # 从新的（带噪）索引重构量化矢量
    #     N, _ = embed_idxs.shape
    #     quant_recon = torch.zeros(N, self.embed_dim, device=embed_idxs.device)
    #     for i in range(self.rq_depth):
    #         embeds = self.codebooks[i].embed(noisy_idxs[:, i])
    #         quant_recon.add_(embeds)

    #     return quant_recon

    def embed(self, noisy_idxs):
        # 4. 从新的（带噪）索引重构量化矢量
        N, _ = noisy_idxs.shape
        quant_recon = torch.zeros(N, self.embed_dim, device=noisy_idxs.device)
        for i in range(self.rq_depth):
            embeds = self.codebooks[i].embed(noisy_idxs[:, i])
            quant_recon.add_(embeds)

        return quant_recon



    def _encode_dynamic_3d(self, x, feat_shape):
        """[内部辅助函数] 动态编码，处理3D可变尺寸输入。"""
        B, L, C = x.shape
        H, W = feat_shape
        assert L == H * W, f"Dynamic input sequence length {L} does not match provided H*W ({H*W})"
        
        # Reshape 3D -> 4D for Conv2d
        x_4d = x.permute(0, 2, 1).view(B, C, H, W).contiguous()
        
        x_4d = self.pre_quant(x_4d)
        x_4d = x_4d.permute(0, 2, 3, 1).contiguous()
        x_flatten = x_4d.reshape(-1, self.embed_dim)
        
        # 保存动态维度信息
        shape_info = (B, H, W)
        return x_flatten, shape_info

    def _decode_dynamic_3d(self, x_quantized_flat, shape_info):
        """[内部辅助函数] 动态解码，使用shape_info恢复3D尺寸。"""
        B, H, W = shape_info
        C = self.latent_shape[-1] # 通道数是固定的

        x_4d = x_quantized_flat.view(B, H, W, self.embed_dim * C)
        x_4d = x_4d.permute(0, 3, 1, 2).contiguous()
        x_4d = self.post_quant(x_4d)
        
        # Reshape 4D -> 3D
        x_3d = x_4d.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return x_3d