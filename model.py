"""
Much of this code is adapted from:
- Andrej Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT)
- MLX adaptation (https://github.com/vithursant/nanoGPT_mlx/tree/main)
- Bark official repo (https://github.com/suno-ai/bark)
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
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


# nanogpt_mlx implementation
class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

    def __call__(self, x, mask, cache=None):
        B, T, C = x.shape
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            key = mx.concatenate([key_cache, key], axis=2)
            value = mx.concatenate([value_cache, value], axis=2)

        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
        mask = mask.reshape(1, 1, T, T)
        att = mx.where(mask[:, :, :T, :T] == 0, att, float("-1e9"))
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (key, value)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.c_fc = nn.Linear(args.n_embd, 4 * args.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * args.n_embd, args.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dropout(self.c_proj(nn.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.ln_1 = nn.LayerNorm(args.n_embd)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd)
        self.mlp = MLP(args)

    def forward(
        self,
        x: mx.array,
        mask,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, cache = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return out, cache


# Bark GPT - Coarse
class GPT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.input_vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.layers = [Block(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, args.output_vocab_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        merge_context: bool = False,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        b, t = x.size()

        if cache is not None:
            assert t == 1
            tok_emb = self.wte(x)
        else:
            if merge_context:
                assert x.shape[1] >= 256 + 256 + 1
                t = x.shape[1] - 256
                tok_emb = mx.concatenate(
                    [
                        self.wte(x[:, :256]) + self.wte(x[:, 256 : 256 + 256]),
                        self.wte(x[:, 256 + 256 :]),
                    ],
                    dim=1,
                )
            else:
                tok_emb = self.wte(x)

        # past length

        pos = mx.arange(0, t, 1, dtype=x.dtype)
        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
        else:
            for block in self.layers:
                x, curr_cache = block(x, mask=mask)
                kv_cache.append(curr_cache)

        x = self.ln_f(x)

        logits = self.lm_head(x[:, [-1], :])

        return logits, cache


class FineGPT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args


def load_model(model_path: str):
    weights = mx.load(model_path)
    weights = tree_unflatten(list(weights.items()))


def generate(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    # Text to Semantic
    # Semantic to Waveform
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bark inference script")
    parser.add_argument("model_path")
    args = parser.parse_args()

    load_model(args.model_path)
    # break up the weights into bark-coarse and bark-fine
    bark_coarse = BarkCoarse(model_args["bark-coarse"])
    print(bark_coarse)
    pass
