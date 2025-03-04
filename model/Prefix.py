# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scaler import Scaler
from .prompt import Prompt


class Prefix(nn.Module):
    def __init__(
        self,
        length: int = 20,
        dim: int = 512,
        position: int = 1,
        key_scale: Optional[float] = None,
        val_scale: Optional[float] = None,
        compensatory: bool = True,
    ):
        super().__init__()

        self.compensatory = compensatory

        args = (length, dim, position, False)
        self.key = Prompt(*args, scale=key_scale)
        self.val = Prompt(*args, scale=val_scale)
        
        #add
        #self.scale_to_grad = 1 #5.0
        self.scale_to_grad = nn.Parameter(torch.tensor(100.0)) #10.0
        print(f"prefix_scale_init: {self.scale_to_grad}")

    def forward(self, key: torch.Tensor, val: torch.Tensor):
        return self.key(key), self.val(val)

    def compensate(self, attn):
        if not self.compensatory:
            return attn

        position, length = self.key.position, self.key.length
        s, t = position, position + length
        lamb = attn[..., s:t].sum(dim=-1, keepdim=True)
        attn1 = attn[..., :s]
        #attn2 = attn[..., s:t] / lamb.clamp(min=1e-12)
        attn2 = attn[..., s:t] * self.scale_to_grad
        attn3 = attn[..., t:]
        attn = torch.cat([attn1, attn2, attn3], dim=-1)

        return attn
