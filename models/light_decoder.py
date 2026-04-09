# coding=utf-8
"""
Lightweight SD2 VAE Decoder with halved channel dimensions.

Original SD2 VAE Decoder:
    block_out_channels = [128, 256, 512, 512]
    layers_per_block = 2  (results in 3 resnets per UpBlock)
    mid_block: 2 ResnetBlock2D + 1 Attention (512-dim)
    conv_in: 4 -> 512
    conv_out: 128 -> 3

Light Decoder (this file):
    block_out_channels = [64, 128, 256, 256]
    layers_per_block = 2  (results in 3 resnets per UpBlock, same structure)
    mid_block: 2 ResnetBlock2D + 1 Attention (256-dim)
    conv_in: 4 -> 256
    conv_out: 64 -> 3

Self-attention in mid_block is preserved. Channel dimensions are halved.
"""

import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D, UNetMidBlock2D


class LightDecoder(nn.Module):
    """
    Lightweight VAE Decoder for SD2 with halved channels.
    
    Accepts latent z of shape [B, 4, H/8, W/8] and outputs RGB image [B, 3, H, W].
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels=(64, 128, 256, 256),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block

        # Reversed block_out_channels for decoder (goes from deepest to shallowest)
        reversed_block_out_channels = list(reversed(block_out_channels))

        # conv_in: latent_channels -> deepest channel dim
        self.conv_in = nn.Conv2d(
            in_channels,
            reversed_block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Mid block with attention
        # attention_head_dim should match in_channels (single-head attention),
        # consistent with diffusers Decoder which uses attention_head_dim=block_out_channels[-1]
        self.mid_block = UNetMidBlock2D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=None,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=reversed_block_out_channels[0],
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        output_channel = reversed_block_out_channels[0]
        for i, up_block_out_channel in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = up_block_out_channel
            is_final_block = i == len(reversed_block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=layers_per_block + 1,  # diffusers convention: layers_per_block + 1 resnets
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)

        # Output layers
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape [B, in_channels, H', W']
        Returns:
            Decoded image of shape [B, out_channels, H, W]
        """
        # conv_in
        h = self.conv_in(z)

        # mid block
        h = self.mid_block(h)

        # up blocks
        for up_block in self.up_blocks:
            h = up_block(h)

        # output
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        return h
