"""
Much of this code is adapted from:
- Andrej Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT)
- MLX adaptation (https://github.com/vithursant/nanoGPT_mlx/tree/main)
- Bark official repo (https://github.com/suno-ai/bark)
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from mlx.utils import tree_unflatten, tree_map

from enum import Enum
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from transformers import BertTokenizer
import glob
import tqdm
import math
from torch_codec import codec_decode
from scipy.io.wavfile import write as write_wav
import torch.nn.functional as F
import torch

mx.random.seed(42)
torch.manual_seed(42)

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
COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050
SAMPLE_RATE = 24_000


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
    def __init__(self, dims: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.bias = mx.zeros((dims,)) if bias else None
        self.weight = mx.ones((dims,))
        self.dims = dims
        self.eps = eps

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
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
        self.bias = (
            mx.tril(mx.ones([args.block_size, args.block_size]))
            .reshape(1, 1, args.block_size, args.block_size)
            .astype(mx.float32)
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

        att = mx.where(
            self.bias[:, :, FULL_T - T : FULL_T, :FULL_T] == 0, float("-1e9"), att
        )
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return (y, present)


class NonCausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

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
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.args = args
        self.ln_1 = LayerNorm(args.n_embd, bias=True)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = LayerNorm(args.n_embd, bias=True)
        self.mlp = MLP(args)
        self.layer_idx = layer_idx

    def __call__(self, x: mx.array, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(
            self.ln_1(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, prev_kvs)


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
        self.drop = nn.Dropout(args.dropout)
        self.layers = [Block(args=args) for _ in range(args.n_layer)]
        self.ln_f = LayerNorm(args.n_embd, bias=True)
        self.lm_head = nn.Linear(args.n_embd, args.output_vocab_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        merge_context: bool = False,
        past_kv: mx.array = None,
        position_ids: mx.array = None,
        use_cache: bool = False,
    ) -> mx.array:
        b, t = x.shape

        if past_kv is not None:
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
                    axis=1,
                )
            else:
                tok_emb = self.wte(x)

        # past length
        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.layers))
        else:
            past_length = past_kv[0][0].shape[-2]

        if position_ids is None:
            position_ids = mx.arange(past_length, t + past_length)
            position_ids = position_ids.reshape(1, -1)  # shape (1, t)

        pos_emb = self.wpe(position_ids)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.layers, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                new_kv = new_kv + (kv,)

        x = self.ln_f(x)

        logits = self.lm_head(
            x[:, -1:, :]
        )  # note: using list [-1] to preserve the time dim

        return (logits, new_kv)


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
        self.drop = nn.Dropout(args.dropout)
        self.layers = [FineBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_heads = [
            nn.Linear(args.n_embd, args.output_vocab_size, bias=False)
            for _ in range(args.n_codes_given, args.n_codes_total)
        ]
        for i in range(self.n_codes_total - args.n_codes_given):
            self.wtes[i + 1].weight = self.lm_heads[i].weight

    def __call__(self, pred_idx: mx.array, idx: mx.array) -> mx.array:
        b, t, codes = idx.shape
        assert (
            t <= self.args.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert pred_idx > 0, "cannot predict 0th codebook"
        assert codes == self.n_codes_total, (b, t, codes)
        pos = mx.arange(0, t).astype(mx.int64).reshape(1, t)  # shape (1, t)
        tok_embs = [
            wte(idx[:, :, i].astype(mx.int32)).reshape(b, t, -1, 1)
            for i, wte in enumerate(self.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = mx.concatenate(tok_embs, axis=-1)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(axis=-1)
        x = self.drop(x + pos_emb)
        for block in self.layers:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_heads[pred_idx - self.args.n_codes_given](x)
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
            bark_coarse.update(weights)
            mx.eval(bark_coarse.parameters())
        elif "fine" in f:
            bark_fine.update(weights)
            mx.eval(bark_fine.parameters())
        elif "text" in f:
            bark_text.update(weights)
            mx.eval(bark_text.parameters())
    return tokenizer, bark_coarse, bark_fine, bark_text


def generate_text_semantic(
    model: nn.Module,
    tokenizer: any,
    text: str,
    temp: float = 0.7,
    use_kv_caching: bool = False,
):
    """Generate semantic tokens from text."""
    print("Generating semantic tokens...")
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
    x = (
        mx.concatenate(
            [encoded_text, semantic_history, mx.array([SEMANTIC_INFER_TOKEN])]
        )
        .reshape(1, -1)
        .astype(mx.int64)
    )
    n_tot_steps = 768
    kv_cache = None
    for i in tqdm.tqdm(range(n_tot_steps)):
        if use_kv_caching and kv_cache is not None:
            x_input = x[:, -1:]
        else:
            x_input = x
        logits, kv_cache = model(
            x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
        )
        relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
        # Early stop
        relevant_logits = mx.concatenate(
            [relevant_logits, logits[0, 0, SEMANTIC_PAD_TOKEN].reshape(1)], axis=-1
        )
        next_token = mx.random.categorical(
            relevant_logits * 1 / (temp), num_samples=1
        ).astype(mx.int32)

        if next_token == SEMANTIC_VOCAB_SIZE:
            print(f"Early stop at step {i} with token {next_token}")
            break
        x = mx.concatenate([x, next_token.reshape(1, -1)], axis=1)
        if i == n_tot_steps - 1:
            break
    out = x.squeeze()[256 + 256 + 1 :]
    return out


def generate_coarse(
    model: nn.Module,
    x_semantic: mx.array,
    temp=0.7,
    silent=False,
    max_coarse_history=60,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    use_kv_caching=False,
):
    """Generate coarse tokens from semantic tokens."""
    print("Generating coarse tokens...")
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(
        math.floor(max_coarse_history / semantic_to_coarse_ratio)
    )
    x_semantic_history = mx.array([], dtype=mx.int32)
    x_coarse_history = mx.array([], dtype=mx.int32)
    n_steps = int(
        round(
            math.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    x_semantic = mx.concatenate([x_semantic_history, x_semantic]).astype(mx.int32)
    x_coarse = x_coarse_history.astype(mx.int32)
    base_semantic_idx = len(x_semantic_history)
    # Inference
    x_semantic_in = x_semantic.reshape(1, -1)
    x_coarse_in = x_coarse.reshape(1, -1)
    n_window_steps = int(round(n_steps / sliding_window_len))
    n_step = 0
    for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
        semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
        x_in = x_semantic_in[:, max(0, semantic_idx - max_semantic_history) :]
        x_in = x_in[:, :256]
        x_in = mx.pad(
            x_in,
            ((0, 0), (0, 256 - x_in.shape[-1])),
            constant_values=COARSE_SEMANTIC_PAD_TOKEN,
        )
        x_in = mx.concatenate(
            [
                x_in,
                mx.array([COARSE_INFER_TOKEN]).reshape(1, -1),
                x_coarse_in[:, -max_coarse_history:],
            ],
            axis=1,
        )
        kv_cache = None
        for _ in range(sliding_window_len):
            if n_step >= n_steps:
                continue
            is_major_step = n_step % N_COARSE_CODEBOOKS == 0
            x_input = x_in[:, -1:] if use_kv_caching and kv_cache is not None else x_in
            logits, kv_cache = model(
                x_input, use_cache=use_kv_caching, past_kv=kv_cache
            )
            logit_start_idx = (
                SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
            )
            logit_end_idx = (
                SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
            )
            logit_end_idx = min(logit_end_idx, logits.shape[-1])
            relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
            item_next = mx.random.categorical(
                relevant_logits * (1 / temp), num_samples=1
            ).astype(mx.int32)

            item_next += logit_start_idx
            x_coarse_in = mx.concatenate([x_coarse_in, item_next.reshape(1, 1)], axis=1)
            x_in = mx.concatenate([x_in, item_next.reshape(1, 1)], axis=1)
            n_step += 1

    gen_coarse_arr = x_coarse_in[0, len(x_coarse_history) :]
    gen_coarse_audio_arr = (
        gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    )
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE

    return gen_coarse_audio_arr


def generate_fine(
    model: nn.Module,
    x_coarse_gen: mx.array,
    temp: float = 0.5,
):
    """Generate fine tokens from coarse tokens."""
    print("Generating fine tokens...")
    x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    in_arr = mx.concatenate(
        [
            x_coarse_gen,
            mx.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ],
        axis=0,
    )
    n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = mx.concatenate(
            [
                in_arr,
                mx.zeros((N_FINE_CODEBOOKS, n_remove_from_end)) + CODEBOOK_SIZE,
            ],
            axis=1,
        )
    # Inference
    n_loops = (
        max(0, int(math.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))) + 1
    )
    in_arr = in_arr.T
    for n in tqdm.tqdm(range(n_loops)):
        start_idx = mx.min(mx.array([n * 512, in_arr.shape[0] - 1024])).item()
        start_fill_idx = mx.min(
            mx.array([n_history + n * 512, in_arr.shape[0] - 512])
        ).item()
        rel_start_fill_idx = start_fill_idx - start_idx
        in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
        for nn in range(n_coarse, N_FINE_CODEBOOKS):
            logits = model(nn, in_buffer)
            if temp is None:
                relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                codebook_preds = mx.argmax(relevant_logits, -1)
            else:
                relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                codebook_preds = (
                    mx.random.categorical(
                        relevant_logits[rel_start_fill_idx:1024], num_samples=1
                    )
                    .reshape(-1)
                    .astype(mx.int32)
                )
            in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
        for nn in range(n_coarse, N_FINE_CODEBOOKS):
            in_arr[
                start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn
            ] = in_buffer[0, rel_start_fill_idx:, nn]
    gen_fine_arr = in_arr.squeeze().T
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    return gen_fine_arr


def generate(text):
    tokenizer, bark_coarse, bark_fine, bark_text = load_model(args.model_path)

    # generate semantic tokens
    semantic_tokens = generate_text_semantic(
        bark_text, tokenizer, text, use_kv_caching=True
    )

    # generate waveform
    coarse_tokens = generate_coarse(
        bark_coarse, x_semantic=semantic_tokens, use_kv_caching=True
    )
    # generate fine codes
    fine_tokens = generate_fine(bark_fine, coarse_tokens, temp=0.5)

    # codec decode
    audio_arr = codec_decode(fine_tokens)

    write_wav("generation.wav", SAMPLE_RATE, audio_arr)

    print("Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bark inference script")
    parser.add_argument("model_path", default="models/")
    parser.add_argument("--text", default="hello world!")
    args = parser.parse_args()

    generate(args.text)
