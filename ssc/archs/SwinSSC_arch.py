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


    def forward(self, input_image, given_SNR=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if self.model == 'SwinJSCC_w/o_SAandRA' or self.model == 'SwinJSCC_w/_SA':
            feature = self.encoder(input_image, chan_param, self.channel_number, self.model)
            CBR = feature.numel() / 2 / input_image.numel()

            x_reshaped, feature_quant, loss_commit, embed_idxs = self.quantizer.ad(feature)

            if self.pass_channel:
                noisy_quant = self.quantizer.feature_pass_channel(embed_idxs, chan_param)
            else:
                noisy_quant = feature_quant

            noisy_quant = x_reshaped + (noisy_quant - x_reshaped).detach()
            feature_dequant = self.quantizer.da(noisy_quant)

        recon_image = self.decoder(feature_dequant, chan_param, self.model)

        return recon_image, CBR, chan_param, loss_commit, embed_idxs

