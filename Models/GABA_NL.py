import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


    
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1)#.contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2)#.contiguous()
        return x
    


class GateLinearAttentionNoSilu(nn.Module):

    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 cross_rank: int = 4, 
                 use_cross_refinement: bool = True, 
                 cross_gamma_init: float = 0.0,
                 vl_scale_max: float = 0.20,
                 eps: float = 1e-6):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        if cross_rank <= 0:
            raise ValueError(f"cross_rank must be positive, but received {cross_rank}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.vl_scale_max = vl_scale_max

        self.scale = self.head_dim ** -0.5
        self.cross_rank = cross_rank
        self.use_cross_refinement = use_cross_refinement
        self.cross_scale = cross_rank ** -0.5

        # Joint Q, K, V, and output-gate projection.
        self.qkvo = nn.Conv2d(dim, dim * 4, kernel_size=1)

        self.elu = nn.ELU()

        # Local positional enhancement applied to V.
        self.lepe = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.gamma_qv = nn.Parameter(torch.zeros(1))
        self.residual_gate = nn.Conv2d(dim, dim, 1)  # MODIFIED:residual token-specific correction branch
        self.residual_proj = nn.Conv2d(dim, dim, 1)  # MODIFIED:residual token-specific correction branch
        # Only one additional convolution.
        self.v_lepe_antisym = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        # Exact original model at initialization.
        nn.init.zeros_(self.v_lepe_antisym.weight)
        # One learnable scale per attention head.
        self.vl_scale_raw = nn.Parameter(torch.full((1, num_heads, 1, 1), -5.0))

        if self.use_cross_refinement:
            self.cross_q_factor = nn.Parameter(torch.empty(num_heads, self.head_dim, cross_rank))
            self.cross_k_factor = nn.Parameter(torch.empty(num_heads, self.head_dim, cross_rank))
            self.cross_output_factor = nn.Parameter(torch.empty(num_heads, cross_rank, self.head_dim))
            self.cross_gamma = nn.Parameter(torch.tensor(float(cross_gamma_init)))

            self.reset_cross_parameters()

    def reset_cross_parameters(self):
        """Initialize the low-rank cross-channel branch."""

        nn.init.xavier_uniform_(self.cross_q_factor)
        nn.init.xavier_uniform_(self.cross_k_factor)
        nn.init.normal_(self.cross_output_factor, mean=1.0, std=0.02)

    def channel_rms(self, feature: torch.Tensor,) -> torch.Tensor:
        """
        Computes RMS across channels.

        Input:
            [B, C, H, W]

        Output:
            [B, 1, H, W]
        """
        return torch.sqrt(feature.square().mean(dim=1, keepdim=True) + self.eps)

    def channel_rms_normalize(self, feature: torch.Tensor) -> torch.Tensor:
        """
        RMS normalization across channels.
        """
        return feature / self.channel_rms(feature)

    def get_channel_scale(self) -> torch.Tensor:
        """
        Converts one scale per head to one scale per channel.

        Output:
            [1, C, 1, 1]
        """
        alpha = self.vl_scale_max * torch.sigmoid(self.vl_scale_raw)
        alpha = alpha.repeat_interleave(self.head_dim, dim=1)

        return alpha


    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        """
        Args:
            x:
                Input tensor of shape (B, C, H, W).

            sin, cos:
                RoPE tensors compatible with theta_shift.

        Returns:
            Tensor of shape (B, C, H, W).
        """

        B, C, H, W = x.shape
        L = H * W

        # Generate Q, K, V, and multiplicative output gate
        qkvo = self.qkvo(x)                     # (B, 4C, H, W)
        qkv = qkvo[:, :3 * self.dim]           # (B, 3C, H, W)
        o = qkvo[:, 3 * self.dim:]             # (B, C, H, W)

        q_map = qkv[:, :self.dim]
        k_map = qkv[:, self.dim:2 * self.dim]
        v_map = qkv[:, 2 * self.dim:]

        
        # V-LePE anti-symmetric
        lepe_base = self.lepe(v_map)
        v_normalized = self.channel_rms_normalize(v_map)
        lepe_normalized = self.channel_rms_normalize(lepe_base)
        contrast = (v_normalized - lepe_normalized).detach()
        direction = self.v_lepe_antisym(contrast)
        direction = torch.tanh(direction)
        shared_magnitude = 0.5 * (self.channel_rms(v_map) + self.channel_rms(lepe_base))
        shared_magnitude = shared_magnitude.detach()
        displacement = direction * shared_magnitude
        alpha = self.get_channel_scale()
        delta = alpha * displacement
        v_guided = v_map + delta
        lepe_guided = lepe_base - delta

        # Original Q, K feature maps
        q = rearrange(q_map, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        k = rearrange(k_map, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        v = rearrange(v_guided, "b (n d) h w -> b n (h w) d", n=self.num_heads)

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = theta_shift(q, sin, cos)
        k_rope = theta_shift(k, sin, cos)
        k_mean = k.mean(dim=-2, keepdim=True,)                         # (B, heads, 1, head_dim)
        denominator = (q @ k_mean.transpose(-2, -1))                   # (B, heads, L, 1)
        z = 1.0 / (denominator + 1e-6)

        # Shared K^T V state
        inv_sqrt_l = L ** -0.5

        kv = (k_rope.transpose(-2, -1) * inv_sqrt_l) @ (v * inv_sqrt_l)
        base_res = (q_rope @ kv) * z   # base_res: (B, heads, L, head_dim)
        q_norm = q / (q.mean(dim=-1, keepdim=True) + 1e-6)
        base_res = base_res + self.gamma_qv * (q_norm * v)

        # Low-rank cross-channel Q-K-V branch
        if self.use_cross_refinement:            
            q_cross = torch.einsum("bhld,hdr->bhlr", q_rope, self.cross_q_factor,) # q_cross: (B, heads, L, rank)
            reduced_kv = torch.einsum("hdr,bhde->bhre", self.cross_k_factor, kv)
            reduced_kv = (reduced_kv * self.cross_output_factor.unsqueeze(0)) # (B, heads, rank, head_dim)
            cross_res = torch.einsum("bhlr,bhre->bhle", q_cross, reduced_kv) # cross_res: (B, heads, L, head_dim)
            cross_res = cross_res * self.cross_scale
            cross_res = cross_res * z
            res = base_res + self.cross_gamma * cross_res

        else:
            res = base_res


        
        res = rearrange(res, "b n (h w) d -> b (n d) h w", h=H, w=W)
        residual_correction = torch.sigmoid(self.residual_gate(x)) * self.residual_proj(x)  # MODIFIED: sigma(W_g X_i) ⊙ U X_i
        res = res + lepe_guided + residual_correction

        return self.proj(res * o)



    
class VanillaSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, vl_scale_max: float = 0.20, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps
        self.vl_scale_max = vl_scale_max
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.residual_gate = nn.Conv2d(dim, dim, 1)  # MODIFIED:residual token-specific correction branch
        self.residual_proj = nn.Conv2d(dim, dim, 1)  # MODIFIED:residual token-specific correction branch
        self.v_lepe_antisym = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        nn.init.zeros_(self.v_lepe_antisym.weight)
        self.vl_scale_raw = nn.Parameter(torch.full((1, num_heads, 1, 1), -5.0))

    def channel_rms(self, feature: torch.Tensor,) -> torch.Tensor:
        """
        Computes RMS across channels.

        Input:
            [B, C, H, W]

        Output:
            [B, 1, H, W]
        """
        return torch.sqrt(feature.square().mean(dim=1, keepdim=True) + self.eps)

    def channel_rms_normalize(self, feature: torch.Tensor) -> torch.Tensor:
        """
        RMS normalization across channels.
        """
        return feature / self.channel_rms(feature)

    def get_channel_scale(self) -> torch.Tensor:
        """
        Converts one scale per head to one scale per channel.

        Output:
            [1, C, 1, 1]
        """
        alpha = self.vl_scale_max * torch.sigmoid(self.vl_scale_raw)
        alpha = alpha.repeat_interleave(self.head_dim, dim=1)
        return alpha

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        
        q_map = qkv[:, :self.dim]
        k_map = qkv[:, self.dim:2 * self.dim]
        v_map = qkv[:, 2 * self.dim:]
        
        # V-LePE anti-symmetric
        lepe_base = self.lepe(v_map)
        v_normalized = self.channel_rms_normalize(v_map)
        lepe_normalized = self.channel_rms_normalize(lepe_base)
        contrast = (v_normalized - lepe_normalized).detach()
        direction = self.v_lepe_antisym(contrast)
        direction = torch.tanh(direction)
        shared_magnitude = 0.5 * (self.channel_rms(v_map) + self.channel_rms(lepe_base))
        shared_magnitude = shared_magnitude.detach()
        displacement = direction * shared_magnitude
        alpha = self.get_channel_scale()
        delta = alpha * displacement
        v_guided = v_map + delta
        lepe_guided = lepe_base - delta

        # Original Q, K feature maps
        q = rearrange(q_map, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        k = rearrange(k_map, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        v = rearrange(v_guided, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        attn = torch.softmax(self.scale * q @ k.transpose(-1, -2), dim=-1) # (b n (h w) (h w))
        res = attn @ v # (b n (h w) d)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        residual_correction = torch.sigmoid(self.residual_gate(x)) * self.residual_proj(x)  # MODIFIED: sigma(W_g X_i) ⊙ U X_i
        res = res + lepe_guided + residual_correction
        return self.proj(res * o)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 activation_fn=F.gelu,
                 dropout=0.0,
                 activation_dropout=0.0,
                 subconv=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, 1)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.dwconv.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class Block(nn.Module):

    def __init__(self, 
                 flag, 
                 embed_dim, 
                 num_heads, 
                 ffn_dim, 
                 drop_path=0., 
                 layerscale=False, 
                 layer_init_value=1e-6):
        
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = LayerNorm2d(embed_dim, eps=1e-6)
        assert flag in ['l', 'v']
        if flag == 'l':
            self.attn = GateLinearAttentionNoSilu(embed_dim, num_heads)
        else:
            self.attn = VanillaSelfAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1),requires_grad=True)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), sin, cos))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), sin, cos))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B C H W
        '''
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        return x
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, 
                 flags, 
                 embed_dim, 
                 out_dim, 
                 depth, 
                 num_heads,
                 ffn_dim=96., 
                 drop_path=0.,
                 downsample: PatchMerging=None,
                 layerscale=False, 
                 layer_init_value=1e-6):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.RoPE = RoPE(embed_dim, num_heads)

        # build blocks
        self.blocks = nn.ModuleList([Block(flags[i], 
                                           embed_dim, 
                                           num_heads, 
                                           ffn_dim, 
                                           drop_path[i] if isinstance(drop_path, list) else drop_path, 
                                           layerscale, 
                                           layer_init_value) for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.size()
        sin, cos = self.RoPE((h, w))
        for blk in self.blocks:
            x = blk(x, sin, cos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
                                  nn.BatchNorm2d(embed_dim//2),
                                  nn.GELU(),
                                  nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
                                  nn.BatchNorm2d(embed_dim//2),
                                  nn.GELU(),
                                  nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
                                  nn.BatchNorm2d(embed_dim//2),
                                  nn.GELU(),
                                  nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
                                  nn.BatchNorm2d(embed_dim),
                                  # nn.GELU(),
                                  # nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                                  # nn.BatchNorm2d(embed_dim)
                                  )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)#(b c h w)
        return x

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen: Tuple[int]):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos
    
class GABA_NL(nn.Module):

    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 flagss=[['l']*10, ['l']*10, ['v']*10, ['v']*10],
                 embed_dims=[96, 192, 384, 768], 
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], 
                 mlp_ratios=[3, 3, 3, 3], 
                 drop_path_rate=0.1, 
                 projection=1024, 
                 layerscales=[False, False, False, False], 
                 layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(flags=flagss[i_layer], 
                               embed_dim=embed_dims[i_layer],
                               out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               layerscale=layerscales[i_layer],
                               layer_init_value=layer_init_values[i_layer])
            self.layers.append(layer)
            
        self.proj = nn.Conv2d(self.num_features, projection, 1)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(projection, num_classes, 1) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x) #(b c h w)
        x = self.norm(x) #(b c h w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1 1
        return x

    def forward(self, x):
        # x = F.interpolate(x, (384, 384), mode='bicubic')
        x = self.forward_features(x)
        x = self.head(x).flatten(1)
        return x
    
@register_model
def GABA_NL_T(args=None):
    model = GABA_NL(embed_dims=[64, 128, 256, 512],
                  depths=[2, 2, 6, 2],
                  num_heads=[1, 2, 4, 8],
                  mlp_ratios=[3.5, 3.5, 3.5, 3.5],
                  drop_path_rate=0.1,
                  projection=1024,
                  layerscales=[True, True, True, True],
                  layer_init_values=[1, 1, 1, 1])
    model.default_cfg = _cfg()
    return model

@register_model
def GABA_NL_S(args=None):
    model = GABA_NL(embed_dims=[64, 128, 320, 512], 
                  depths=[3, 5, 9, 3],
                  num_heads=[1, 2, 5, 8],
                  mlp_ratios=[3.5, 3.5, 3.5, 3.5],
                  drop_path_rate=0.15,
                  projection=1024,
                  layerscales=[True, True, True, True],
                  layer_init_values=[1, 1, 1, 1])
    model.default_cfg = _cfg()
    return model

@register_model
def GABA_NL_B(args=None):
    model = GABA_NL(embed_dims=[96, 192, 384, 512],
                  depths=[4, 6, 12, 6],
                  num_heads=[1, 2, 6, 8],
                  mlp_ratios=[3.5, 3.5, 3.5, 3.5],
                  drop_path_rate=0.4,
                  projection=1024,
                  layerscales=[True, True, True, True],
                  layer_init_values=[1, 1, 1e-6, 1e-6])
    model.default_cfg = _cfg()
    return model

@register_model
def GABA_NL_L(args=None):
    model = GABA_NL(embed_dims=[96, 192, 448, 640], 
                  depths=[4, 7, 19, 8],
                  num_heads=[1, 2, 7, 10],
                  mlp_ratios=[3.5, 3.5, 3.5, 3.5],
                  drop_path_rate=0.55,
                  projection=1024,
                  layerscales=[True, True, True, True],
                  layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6])
    model.default_cfg = _cfg()
    return model