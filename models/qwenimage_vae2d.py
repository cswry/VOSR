# Copyright 2025
# The Qwen-Image Team, Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.activations import get_activation
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution


class RMSNorm2D(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, bias: bool = False):
        super().__init__()
        assert channel_first, "Only channel_first=True is supported."
        self.channel_first = True
        self.scale = dim ** 0.5
        shape = (dim, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class ResidualBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = RMSNorm2D(in_dim, channel_first=True)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = RMSNorm2D(out_dim, channel_first=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h


class AttentionBlock2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, 3, c, h * w)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()

        q, k, v = qkv.unbind(dim=1)
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.squeeze(1)

        x_attn = x_attn.permute(0, 2, 1).reshape(b, c, h, w)
        x_attn = self.proj(x_attn)
        return x_attn + identity


class Resample2D(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class MidBlock2D(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim
        resnets = [ResidualBlock2D(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock2D(dim))
            resnets.append(ResidualBlock2D(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x)
        return x


class Encoder2D(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.nonlinearity = get_activation(non_linearity)

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = nn.Conv2d(3, dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock2D(in_dim, out_dim, dropout, non_linearity))
                if scale in attn_scales:
                    self.down_blocks.append(AttentionBlock2D(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                self.down_blocks.append(Resample2D(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = MidBlock2D(out_dim, dropout, non_linearity, num_layers=1)

        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


class UpBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float = 0.0, upsample_mode: Optional[str] = None, non_linearity: str = "silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock2D(cur, out_dim, dropout, non_linearity))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([Resample2D(out_dim, mode=upsample_mode)])

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Decoder2D(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.nonlinearity = get_activation(non_linearity)

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv_in = nn.Conv2d(z_dim, dims[0], kernel_size=3, padding=1)
        self.mid_block = MidBlock2D(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample2d"
            up_block = UpBlock2D(in_dim, out_dim, num_res_blocks, dropout, upsample_mode, non_linearity)
            self.up_blocks.append(up_block)

        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


class AutoencoderKLQwenImage2D(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    2D Qwen/Wan VAE for image-only use cases.
    API: encode(x) -> latent_dist, decode(z) -> sample, forward(sample) -> reconstruct.
    """
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = None,
        latents_std: List[float] = None,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        self.encoder = Encoder2D(
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            dropout=dropout,
            non_linearity=non_linearity,
        )
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, kernel_size=1)

        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, kernel_size=1)
        self.decoder = Decoder2D(
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            dropout=dropout,
            non_linearity=non_linearity,
        )

        if latents_mean is None:
            latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        if latents_std is None:
            latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        self.use_slicing = False
        self.use_tiling = False

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        x = self.post_quant_conv(z)
        dec = self.decoder(x)
        dec = torch.clamp(dec, -1.0, 1.0)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec
