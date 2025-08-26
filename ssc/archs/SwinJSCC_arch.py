from .decoder import *
from .encoder import *
from .channel import Channel
from loss.distortion import Distortion
from random import choice
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SwinJSCC(nn.Module):
    def __init__(self, args, encoder_kwargs, decoder_kwargs, **kwargs):
        super(SwinJSCC, self).__init__()
        self.encoder = SwinJSCC_Encoder(**encoder_kwargs)
        self.decoder = SwinJSCC_Decoder(**decoder_kwargs)

        self.distortion_loss = Distortion(args)
        self.channel = Channel(args)
        self.pass_channel = args.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = [int(snr) for snr in args.multiple_snr.split(",")]
        self.channel_number = [int(c) for c in args.C.split(",")]
        self.downsample = args.downsample
        self.model = args.model
        self.norm = args.norm

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR=None, given_rate=None):
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

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        if self.model == 'SwinJSCC_w/o_SAandRA' or self.model == 'SwinJSCC_w/_SA':
            feature = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == 'SwinJSCC_w/_RA' or self.model == 'SwinJSCC_w/_SAandRA':
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

