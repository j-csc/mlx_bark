# Copyright © 2023 Apple Inc.

import argparse
from itertools import starmap

import numpy as np
import torch
import glob


def weight_mapping(state, model_size):
    # there's no _orig_mod.transformer
    state = {k.replace("_orig_mod.transformer.", ""): v for k, v in state.items()}
    # transformer block mapping
    layer_count = 24 if model_size == "large" else 12
    for i in range(layer_count):
        prefix = f"h.{i}."
        state = {k.replace(prefix, f"layers.{i}."): v for k, v in state.items()}
    # lm_head
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Bark weights to MLX")
    parser.add_argument("--torch_weights_dir", default="weights/")
    parser.add_argument("--model", default="large", choices=["large", "small"])
    args = parser.parse_args()
    if args.model == "large":
        file_pattern = f"{args.torch_weights_dir}*_2.pt"
        weights = glob.glob(file_pattern)
    else:
        all_files = glob.glob(f"{args.torch_weights_dir}*.pt")
        weights = [file for file in all_files if "_2" not in file]
    for w in weights:
        state = torch.load(w, map_location=torch.device("cpu"))
        state = weight_mapping(state["model"], args.model)
        print(state)
        np.savez(
            w.replace(".pt", ".npz"),
            **{
                k: v.numpy()
                for k, v in starmap(lambda k, v: (k, v.squeeze().cpu()), state.items())
            },
        )
