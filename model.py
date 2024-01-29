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
from transformers import BertTokenizer
import glob
import torch

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129595
SEMANTIC_INFER_TOKEN = 129_599

CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000


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


class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float, bias: bool = True):
        super().__init__()
        self.bias = mx.zeros((dims,)) if bias else None
        self.weight = mx.ones((dims,))
        self.dims = dims
        self.eps = eps

    def __call__(self, x):
        mean = mx.mean(axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        if self.bias is not None:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight
        return x


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
        self.bias = mx.tril(mx.ones([args.block_size, args.block_size])).reshape(
            1, 1, args.block_size, args.block_size
        )

    def __call__(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        if past_kv is not None:
            past_key, past_value = past_kv
            key = mx.concatenate([past_key, key], axis=-2)
            value = mx.concatenate([past_value, value], axis=-2)
        FULL_T = key.shape[-2]
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
        inf_mask = mx.array(
            self.bias[:, :, FULL_T - T : FULL_T, :FULL_T] == 0, dtype=mx.float32
        ) * float("-inf")
        att = mx.where(inf_mask, att, inf_mask)
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return (y, present)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)


class NonCausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        # self.dropout = args.dropout

    def __call__(self, x):
        B, T, C = x.shape
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.c_fc = nn.Linear(args.n_embd, 4 * args.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * args.n_embd, args.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dropout(self.c_proj(nn.gelu(self.c_fc(x))))
        # return self.c_proj(nn.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.args = args
        self.ln_1 = nn.LayerNorm(args.n_embd)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd)
        self.mlp = MLP(args)
        self.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        mask,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, cache = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return (out, cache)


class FineBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.ln_1 = nn.LayerNorm(args.n_embd)
        self.attn = NonCausalSelfAttention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd)
        self.mlp = MLP(args)

    def __call__(self, x: mx.array):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.input_vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        # self.drop = nn.Dropout(args.dropout)
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
        b, t = x.shape

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
        if cache is None:
            past_length = 0
            cache = tuple([None] * len(self.layers))
        else:
            past_length = cache[0][0].size(-2)

        if position_ids is None:
            position_ids = mx.arange(past_length, t + past_length)
            position_ids = position_ids.reshape(1, -1)  # shape (1, t)

        pos_emb = self.wpe(position_ids)  # position embeddings of shape (1, t, n_embd)

        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])
        tok_emb = self.wte(x)
        x = tok_emb + pos_emb
        # x = self.drop(tok_emb + pos_emb)

        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache = self.layers[i](x, mask=None, cache=cache[i])
        else:
            for block in self.layers:
                (x, curr_cache) = block(x, mask=mask)
                kv_cache.append(curr_cache)

        x = self.ln_f(x)

        logits = self.lm_head(x[:, [-1], :])

        return logits, cache


class FineGPT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_codes_total = args.n_codes_total
        self.wtes = [
            nn.Embedding(args.input_vocab_size, args.n_embd)
            for _ in range(args.n_codes_total)
        ]
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        # self.drop = nn.Dropout(args.dropout)
        self.layers = [FineBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_heads = [
            nn.Linear(args.n_embd, args.output_vocab_size, bias=False)
            for _ in range(args.n_codes_total)
        ]
        for i in range(self.n_codes_total - args.n_codes_given):
            self.wtes[i + 1].weight = self.lm_heads[i].weight

    def __call__(self, pred_idx: mx.array, idx: mx.array) -> mx.array:
        b, t, codes = idx.shape
        pos = mx.arange(0, t).unsqueeze(0)
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1) for i, wte in enumerate(self.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = mx.cat(tok_embs, dim=-1)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = x + pos_emb
        # x = self.drop(x + pos_emb)
        for block in self.layers:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits


def load_model(model_dir: str):
    # break up the weights into bark-coarse and bark-fine
    files = glob.glob(f"{model_dir}*.npz")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bark_text = GPT(model_args["bark-coarse"])
    bark_fine = FineGPT(model_args["bark-fine"])
    bark_coarse = GPT(model_args["bark-coarse"])
    for f in files:
        weights = mx.load(str(f))
        weights = tree_unflatten(list(weights.items()))
        if "coarse" in f:
            for name, weight in weights.items():
                if hasattr(bark_coarse, name):
                    setattr(bark_coarse, name, weight)
                else:
                    print(f"Weight for {name} not found in the model.")
            bark_coarse.update(weights)
        elif "fine" in f:
            for name, weight in weights.items():
                if hasattr(bark_fine, name):
                    setattr(bark_fine, name, weight)
                else:
                    print(f"Weight for {name} not found in the model.")
            bark_fine.update(weights)
        elif "text" in f:
            for name, weight in weights.items():
                if hasattr(bark_text, name):
                    setattr(bark_text, name, weight)
                else:
                    print(f"Weight for {name} not found in the model.")
            bark_text.update(weights)
    mx.eval(bark_coarse.parameters())
    mx.eval(bark_fine.parameters())
    mx.eval(bark_text.parameters())
    return tokenizer, bark_coarse, bark_fine, bark_text


def generate(
    model: nn.Module,
    text: str,
    temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    # Text to Semantic
    # Semantic to Waveform
    pass


def generate_text_semantic(
    model: nn.Module, tokenizer: any, text: str, temp: float = 0.7, silent: bool = False
):
    encoded_text = (
        mx.array(tokenizer.encode(text, add_special_tokens=False))
        + TEXT_ENCODING_OFFSET
    )
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        encoded_text = encoded_text[:256]
    encoded_text = mx.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
    )
    semantic_history = mx.array([SEMANTIC_PAD_TOKEN] * 256)
    x = mx.concatenate(
        [encoded_text, semantic_history, mx.array([SEMANTIC_INFER_TOKEN])]
    ).reshape(1, -1)

    n_tot_steps = 768
    cache = None
    for i in range(n_tot_steps):
        # look at using cache
        if cache:
            x = x[:, [-1]]
        logits, cache = model(x)
        logits = logits[:, -1, :] / temp
        relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
        probs = mx.softmax(relevant_logits)
        next_token = mx.random.multinomial(probs, dtype=mx.int32)
        x = mx.concatenate([x, next_token.reshape(1, 1)], axis=1)
    out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bark inference script")
    parser.add_argument("model_path")
    args = parser.parse_args()

    # tokenizer, bark_coarse, bark_fine, bark_text = load_model(args.model_path)

    bark_coarse = GPT(model_args["bark-coarse"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bark_text = GPT(model_args["bark-coarse"])

    # generate semantic tokens
    generate_text_semantic(bark_text, tokenizer, "hello world")

    # generate waveform

    pass
