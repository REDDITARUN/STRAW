"""ModulatorNetTRM -- hypernetwork with TRM-style recursive encoder + RoPE.

This variant replaces the simple CNN style encoder with a token-based encoder
that performs K recursive refinement steps using a tiny Transformer block.

Design goals:
- Add compute via recursion (TRM-style) without blowing up parameter count.
- Use RoPE in attention instead of learned absolute positional embeddings.
- Keep the rest of the hypernetwork (low-rank weight reconstruction + vmap)
  identical to ModulatorNet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key.

    Expected shapes:
    - q, k: [B, H, L, D]
    - cos, sin: [L, D]
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    # Broadcast cos/sin over batch + heads
    cos_ = cos[None, None, :, :]
    sin_ = sin[None, None, :, :]
    q_embed = (q * cos_) + (_rotate_half(q) * sin_)
    k_embed = (k * cos_) + (_rotate_half(k) * sin_)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Classic RoPE cache (cos/sin) for a fixed maximum sequence length."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, D]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class TinyAttention(nn.Module):
    """Minimal attention with optional RoPE on Q/K."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        # x: [B, L, D]
        b, l, d = x.shape
        qkv = self.qkv(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, L, Hd]
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos[:l], sin[:l])

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, H, L, Hd]
        attn = attn.transpose(1, 2).contiguous().view(b, l, d)
        return self.out(attn)


class TinyTransformerBlock(nn.Module):
    """Tiny transformer block: RMSNorm -> Attn -> RMSNorm -> MLP."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = TinyAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cos_sin))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TRMStyleEncoder(nn.Module):
    """TRM-style recursive encoder for images.

    Converts an image to a small sequence of tokens and performs K recursive
    refinement steps with a tiny transformer block. Outputs a 64-d style vector.
    """

    def __init__(
        self,
        token_dim: int = 64,
        num_heads: int = 4,
        steps: int = 4,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.steps = steps

        # Stem -> (B, token_dim, 7, 7) for 28x28 inputs
        self.stem = nn.Sequential(
            nn.Conv2d(1, token_dim, 3, 2, 1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(token_dim, token_dim, 3, 2, 1),  # 7x7
            nn.ReLU(),
        )

        self.seq_len = 7 * 7
        self.rotary = RotaryEmbedding(
            dim=token_dim // num_heads,
            max_position_embeddings=self.seq_len,
            base=rope_theta,
        )

        self.block = TinyTransformerBlock(
            hidden_size=token_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Learnable initial latent state z0
        self.z0 = nn.Parameter(torch.zeros(1, self.seq_len, token_dim))
        nn.init.normal_(self.z0, std=0.02)

        # Pool -> style vector (64-d, bounded)
        self.to_style = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        x = self.stem(images)  # [B, D, 7, 7]
        x = x.flatten(2).transpose(1, 2)  # [B, 49, D]

        cos_sin = self.rotary()
        z = self.z0.expand(b, -1, -1)

        # TRM-style recursion: do (steps-1) without grad, final step with grad
        if self.steps <= 1:
            z = self.block(z + x, cos_sin)
        else:
            with torch.no_grad():
                for _ in range(self.steps - 1):
                    z = self.block(z + x, cos_sin)
                    z = self.block(z, cos_sin)
            z = self.block(z + x, cos_sin)
            z = self.block(z, cos_sin)

        pooled = z.mean(dim=1)  # [B, D]
        return self.to_style(pooled)  # [B, 64]


class ModulatorNetTRM(nn.Module):
    """Hypernetwork that uses TRM+RoPE encoder before low-rank weight generation."""

    def __init__(
        self,
        executor_model: nn.Module,
        rank: int = 16,
        *,
        trm_token_dim: int = 64,
        trm_heads: int = 4,
        trm_steps: int = 4,
        trm_mlp_ratio: float = 4.0,
        trm_rope_theta: float = 10000.0,
        trm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.executor_model = executor_model
        self.rank = rank

        # Freeze executor template weights (generated params are used in forward).
        for p in self.executor_model.parameters():
            p.requires_grad = False

        self.style_encoder = TRMStyleEncoder(
            token_dim=trm_token_dim,
            num_heads=trm_heads,
            steps=trm_steps,
            mlp_ratio=trm_mlp_ratio,
            rope_theta=trm_rope_theta,
            dropout=trm_dropout,
        )

        # Compute low-rank parameter budget for every executor parameter
        self.target_shapes = {k: v.shape for k, v in executor_model.named_parameters()}
        self.layer_configs = []
        total_params = 0

        for name, shape in self.target_shapes.items():
            if len(shape) > 1:
                out_d = shape[0]
                in_d = shape[1:].numel()
                count = (out_d * rank) + (rank * in_d)
                self.layer_configs.append(
                    {"name": name, "type": "w", "shape": shape, "dims": (out_d, in_d), "count": count}
                )
            else:
                count = shape.numel()
                self.layer_configs.append({"name": name, "type": "b", "shape": shape, "count": count})
            total_params += count

        self.generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
        )

    def get_weights_for_single_sample(self, style_vector: torch.Tensor):
        flat_params = self.generator(style_vector)
        weights = {}
        curr = 0

        for config in self.layer_configs:
            count = config["count"]
            chunk = flat_params[curr : curr + count]
            curr += count

            if config["type"] == "w":
                out_d, in_d = config["dims"]
                size_a = out_d * self.rank
                mat_a = chunk[:size_a].view(out_d, self.rank)
                mat_b = chunk[size_a:].view(self.rank, in_d)
                w = torch.matmul(mat_a, mat_b)
                weights[config["name"]] = w.view(config["shape"])
            else:
                weights[config["name"]] = chunk.view(config["shape"])

        return weights

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        style_vectors = self.style_encoder(images)

        def executor_fwd_pass(params, single_input):
            return functional_call(self.executor_model, params, single_input.unsqueeze(0)).squeeze(0)

        batch_weights = vmap(self.get_weights_for_single_sample)(style_vectors)
        outputs = vmap(executor_fwd_pass, randomness="different")(batch_weights, images)
        return outputs

