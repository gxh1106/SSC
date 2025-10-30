from .decoder import *
from .encoder import *
from .quantizations import RQBottleneck
from random import choice
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SwinSSC(nn.Module):
    def __init__(self, args, encoder_kwargs, decoder_kwargs, rq_kwargs, **kwargs):
        super(SwinSSC, self).__init__()
        self.encoder = SwinJSCC_Encoder(**encoder_kwargs)
        self.decoder = SwinJSCC_Decoder(**decoder_kwargs)

        self.pass_channel = args.pass_channel
        self.H = self.W = 0
        self.multiple_snr = [int(snr) for snr in args.multiple_snr.split(",")]
        self.channel_number = int(args.C)
        self.downsample = args.downsample
        self.model = args.model
        self.norm = args.norm

        self.quantizer = RQBottleneck(**rq_kwargs)

    def freeze_endec(self):
        """Freezing encoder and decoder parameters."""
        for name, param in self.encoder.named_parameters():
            # if 'head_list' in name:
            #     continue
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_endec(self):
        """Unfreezing encoder and decoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
            
    def forward(self, input_image, given_SNR=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            H_feat, W_feat = H // (2 ** self.downsample), W // (2 ** self.downsample)
            self.decoder.update_resolution(H_feat, W_feat)
            self.H = H
            self.W = W
        else:
            # 如果尺寸未变，也需要计算一次
            H_feat, W_feat = H // (2 ** self.downsample), W // (2 ** self.downsample)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, chan_param, self.channel_number, self.model)
        # CBR = feature.numel() / 2 / input_image.numel()
        CBR = self.quantizer.embed_dim * self.quantizer.rq_depth / (H * W * 3)

        feat_shape = (H_feat, W_feat) if not self.training else None
        x_reshaped, feature_quant, loss_commit, embed_idxs, shape_info = self.quantizer.ad(feature, feat_shape=feat_shape)

        if self.pass_channel:
            noisy_quant = self.quantizer.feature_pass_channel(embed_idxs, chan_param)
        else:
            noisy_quant = feature_quant

        noisy_quant = x_reshaped + (noisy_quant - x_reshaped).detach()
        feature_dequant = self.quantizer.da(noisy_quant, shape_info)

        # loss_commit += (feature_dequant - feature).pow(2.0).mean()

        recon_image = self.decoder(feature_dequant, chan_param, self.model)

        return recon_image, CBR, chan_param, loss_commit, embed_idxs
    
    def forward_faim(self, input_image, given_SNR=None, channel=None, idx_H=0):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            H_feat, W_feat = H // (2 ** self.downsample), W // (2 ** self.downsample)
            self.decoder.update_resolution(H_feat, W_feat)
            self.H = H
            self.W = W
        else:
            # 如果尺寸未变，也需要计算一次
            H_feat, W_feat = H // (2 ** self.downsample), W // (2 ** self.downsample)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, chan_param, self.channel_number, self.model)
        # CBR = feature.numel() / 2 / input_image.numel()
        CBR = H_feat * W_feat * self.quantizer.embed_dim * self.quantizer.rq_depth / (H * W * 3 * 8)

        feat_shape = (H_feat, W_feat) if not self.training else None
        x_reshaped, feature_quant, loss_commit, embed_idxs, shape_info = self.quantizer.ad(feature, feat_shape=feat_shape)

        if self.pass_channel:
            error_config = {'target_layer': idx_H, 'noise_factor': 10000} # L0层加100倍的噪声
            # noisy_quant = self.quantizer.feature_pass_channel(embed_idxs, chan_param, noise_config=error_config)
            # noisy_quant = self.quantizer.feature_pass_channel(embed_idxs, chan_param)
            noisy_quant = self.quantizer.feature_pass_error(embed_idxs, chan_param, noise_config=error_config)

            # noisy_idxs = channel(embed_idxs, chan_param, idx_H) # 默认选择第0层采用端口索引传输
            # noisy_idxs = channel(embed_idxs, chan_param, idx_H, ssc=True, ssc_idx=0) # 选择第ssc_idx层采用端口索引传输
            # noisy_idxs = channel(embed_idxs, chan_param, idx_H, ssc=True, ssc_adapt=True) # 自适应索引分流
            # noisy_idxs = channel(embed_idxs, chan_param, idx_H, ssc=False) # 无非均等保护，EEP方案
            # noisy_quant = self.quantizer.embed(noisy_idxs)
        else:
            noisy_quant = feature_quant

        feature_dequant = self.quantizer.da(noisy_quant, shape_info)

        recon_image = self.decoder(feature_dequant, chan_param, self.model)

        return recon_image, CBR, chan_param, loss_commit, embed_idxs

