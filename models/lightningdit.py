"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from .swiglu_ffn import SwiGLUFFN 
from .pos_embed import VisionRotaryEmbeddingFast
from .rmsnorm import RMSNorm



class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, fused_attn: bool = True, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5  
        self.fused_attn = fused_attn

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        B_cond, N_cond, _ = cond.shape

        q = self.q_linear(x)
        k = self.k_linear(cond)
        v = self.v_linear(cond)


        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
    
        q = self.q_norm(q)
        k = self.k_norm(k)


        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v


        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x





@torch.compile
def modulate_adasin(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale) + shift
@torch.compile
def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    # @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        z_dims=None,
        num_fused_layers=1,
        encdim_ratio=2,
        **block_kwargs
    ):
        super().__init__()
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )


        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.z_dims = z_dims
        if self.z_dims is not None:
            # self.num_fused_layers = num_fused_layers
            # self.layer_weights = nn.Parameter(torch.zeros(num_fused_layers))
            # approx_gelu = lambda: nn.GELU(approximate="tanh")
            # self.layer_norm = nn.LayerNorm(self.z_dims)
            # self.mlp_ca = Mlp(
            #     in_features=self.z_dims,
            #     hidden_features=hidden_size * encdim_ratio,
            #     out_features=hidden_size,
            #     act_layer=approx_gelu,
            #     drop=0
            # )

            self.cross_attn = MultiHeadCrossAttention(d_model=hidden_size, num_heads=num_heads, qk_norm=use_qknorm)
            # self.cross_attn = MultiHeadCrossAttention(d_model=hidden_size, num_heads=num_heads, qk_norm=use_qknorm, **block_kwargs)

    @torch.compile
    def forward(self, x, c, z=None, feat_rope=None):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        x = x + gate_msa * self.attn(modulate_adasin(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        if self.z_dims is not None:
            x = x + self.cross_attn(x, z)
        x = x + gate_mlp * self.mlp(modulate_adasin(self.norm2(x), shift_mlp, scale_mlp))

        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        out_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        z_dims=None,
        encdim_ratio=2, 
        num_fused_layers=1,
        auxiliary_time_cond=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels  # lq + zt
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True, strict_img_size=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     z_dims=z_dims,
                     num_fused_layers=num_fused_layers,
                     encdim_ratio=encdim_ratio,
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.num_fused_layers = num_fused_layers
            # self.layer_weights = nn.Parameter(torch.zeros(num_fused_layers))
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.layer_norm = nn.LayerNorm(self.z_dims)
            self.mlp_ca = Mlp(
                in_features=self.z_dims,
                hidden_features=hidden_size * encdim_ratio,
                out_features=hidden_size,
                act_layer=approx_gelu,
                drop=0
            )
            
        self.auxiliary_time_cond = auxiliary_time_cond
        if auxiliary_time_cond:
            self.r_embedder = TimestepEmbedder(hidden_size)
        else:
            self.r_embedder = None

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.auxiliary_time_cond:
            nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)


        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def enable_fused_attn(self):
        for block in self.blocks:
            block.attn.fused_attn = True
            block.cross_attn.fused_attn = True
            
    
    def disable_fused_attn(self):
        for block in self.blocks:
            block.attn.fused_attn = False
            block.cross_attn.fused_attn = False

    def forward(self, x, t, r=None, z=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of auxiliary timesteps (shortcut endpoint)
        """

        use_checkpoint = self.use_checkpoint
        x = self.x_embedder(x)
        t_raw = t                                                        # (N,) save raw timestep
        t = self.t_embedder(t)                                          # (N, D)
        r = self.r_embedder(r)*(t_raw-r).unsqueeze(-1) if self.r_embedder is not None else 0    # (N, D)
        c = t + r                                                        # (N, D)

        c0 = self.t_block(c)

        if self.z_dims is not None:
            z = z[0]
            z = self.layer_norm(z)  
            z = self.mlp_ca(z)  

        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c0, z, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c0, z, self.feat_rope)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x
    
    def _get_dynamic_rope(self, hw_seq_len, device, dtype):
        """
        Dynamically generate RoPE for variable-size inputs.

        Args:
            hw_seq_len: patch grid side length (H // patch_size)
            device: target device
            dtype: target dtype
        Returns:
            A temporary RoPE forward callable.
        """
        if not self.use_rope:
            return None

        half_head_dim = self.hidden_size // self.num_heads // 2
        pt_seq_len = self.x_embedder.img_size[0] // self.patch_size

        from einops import repeat
        theta = 10000
        freqs = 1. / (theta ** (torch.arange(0, half_head_dim, 2, device=device)[:(half_head_dim // 2)].float() / half_head_dim))

        t = torch.arange(hw_seq_len, device=device).float() / hw_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        from .pos_embed import broadcat
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).to(dtype)
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).to(dtype)

        def dynamic_rope_fn(t_input):
            from .pos_embed import rotate_half
            return t_input * freqs_cos + rotate_half(t_input) * freqs_sin

        return dynamic_rope_fn

    def forward_flexible(self, x, t, r=None, z=None):
        """
        Forward pass that supports variable input sizes.
        Unlike the standard forward, this dynamically generates RoPE
        to accommodate the current input resolution.
        """
        use_checkpoint = self.use_checkpoint

        N, C, H, W = x.shape
        assert H == W, "forward_flexible currently only supports square inputs"

        current_hw_seq_len = H // self.patch_size

        x = self.x_embedder(x)   # (N, T, D)
        t_raw = t   
        t = self.t_embedder(t)                                          # (N, D)
        r = self.r_embedder(r)*(t_raw-r).unsqueeze(-1) if self.r_embedder is not None else 0    # (N, D)
        c = t + r                                                        # (N, D)

        c0 = self.t_block(c)

        if self.z_dims is not None and z is not None:
            z = z[0]
            z = self.layer_norm(z)  
            z = self.mlp_ca(z)

        if self.use_rope:
            train_hw_seq_len = self.x_embedder.img_size[0] // self.patch_size
            if current_hw_seq_len != train_hw_seq_len:
                feat_rope = self._get_dynamic_rope(current_hw_seq_len, x.device, x.dtype)
            else:
                feat_rope = self.feat_rope
        else:
            feat_rope = None

        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c0, z, feat_rope, use_reentrant=True)
            else:
                x = block(x, c0, z, feat_rope)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, H, W)

        return x

  

def interpolate_pos_embed_2d(pos_embed, new_size, old_size):
    """
    Interpolate 2D positional embeddings via bicubic resize.
    pos_embed: [1, old_H*old_W, D]
    new_size: (new_H, new_W)
    old_size: (old_H, old_W)
    """
    if new_size == old_size:
        return pos_embed

    old_H, old_W = old_size
    new_H, new_W = new_size
    D = pos_embed.shape[-1]

    pos_embed_2d = pos_embed.reshape(1, old_H, old_W, D).permute(0, 3, 1, 2)

    pos_embed_new = F.interpolate(
        pos_embed_2d,
        size=(new_H, new_W),
        mode='bicubic',
        align_corners=False
    )

    pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(1, new_H * new_W, D)

    return pos_embed_new

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

