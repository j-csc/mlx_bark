import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
from mlx.utils import tree_unflatten

from enum import Enum
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math


class Keys(Enum):
    semantic = "semantic"
    coarse_acoustics = "coarse_acoustics"
    codec_model = "codec_model"
    fine_acoustics = "fine_acoustics"


@dataclass
class ModelArgs:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    n_codes_total: Optional[int] = None
    n_codes_given: Optional[int] = None


model_args = {
    "bark-coarse": ModelArgs(),
    "bark-fine": ModelArgs(n_codes_given=1, n_codes_total=8),
}


# Non-causal attention, with bias to prevent attention to future tokens
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = nn.Dropout(args.dropout)
        self.bias = mx.tril(mx.ones((args.block_size, args.block_size))).reshape(
            1, 1, args.block_size, args.block_size
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, T, C = x.size()

        queries, keys, values = self.c_attn(x).split(self.n_embd, dim=2)
        queries, keys, values = [
            z.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            for z in (queries, keys, values)
        ]  # (B, nh, T, hs)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], dim=-2)
            values = mx.concatenate([value_cache, values], dim=-2)

        FULL_T = keys.shape[-2]  # full length of the sequence
        scale = 1.0 / math.sqrt(keys.size(-1))

        scores = (queries @ keys.transpose(-2, -1)) * scale
        mask = self.bias[:, :, FULL_T - T : FULL_T, :FULL_T] == 0
        scores = mx.where(mask, float("-inf"), scores)
        scores = mx.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        out = scores @ values  # (Bs, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        out = self.resid_dropout(self.c_proj(out))
        return out, (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.n_embd, 4 * args.n_embd, bias=False)
        self.w2 = nn.Linear(4 * args.n_embd, args.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dropout(self.w2(nn.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.ln_1 = nn.LayerNorm(args.n_embd)
        self.attention = Attention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd)
        self.feed_forward = FeedForward(args)

    def forward(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, cache = self.attention(self.ln_1(x), cache)
        h = x + r
        r = self.feed_forward(self.ln_2(h))
        out = h + r
        return out, cache


class BarkCoarse(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.input_vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, args.output_vocab_size, bias=False)


class BarkFine(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args


def load_model(model_path: str):
    weights = mx.load(model_path)
    weights = tree_unflatten(list(weights.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bark inference script")
    parser.add_argument("model_path")
    args = parser.parse_args()

    # load_model(args.model_path)
    model = BarkCoarse(model_args["bark-coarse"])
    print(model)
    pass
