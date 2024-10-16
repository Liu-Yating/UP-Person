import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F



class MultiheadAttention(Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        
        self.in_proj_weight = nn.Parameter(torch.empty((3 * dim, dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
        self.out_proj = nn.Linear(dim, dim, bias=True)
        

    def forward(self, x, register_hook=False, prompt=None):
        N, B, C = x.shape
        qkv = (x @ self.in_proj_weight.T).reshape(N, B, 3, self.num_heads, C // self.num_heads).permute(2, 1, 3, 0, 4) # qkv, B, H, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).permute(1, 0, 2)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x
        
    