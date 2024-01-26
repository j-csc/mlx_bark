# Copyright © 2023 Apple Inc.

import argparse
from itertools import starmap

import numpy as np
import torch
import glob

# change
# MLP = Feedforward
# Block = TransformerBlock
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Bark weights to MLX")
    parser.add_argument("torch_weights_dir")
    args = parser.parse_args()
    weights = glob.glob(f"{args.torch_weights_dir}*.pt")
    for w in weights:
        state = torch.load(w, map_location=torch.device("cpu"))["model"]
        print(state.keys())
        np.savez(
            w.replace(".pt", ".npz"),
            **{
                k: v.numpy()
                for k, v in starmap(lambda k, v: (k, v.squeeze().cpu()), state.items())
            },
        )
