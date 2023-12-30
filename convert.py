# Copyright © 2023 Apple Inc.

import argparse
from itertools import starmap

import numpy as np
import torch

# change
# MLP = Feedforward
# Block = TransformerBlock
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Bark weights to MLX")
    parser.add_argument("torch_weights")
    parser.add_argument("output_file")
    args = parser.parse_args()

    state = torch.load(args.torch_weights)
    np.savez(
        args.output_file,
        **{
            k: v.numpy()
            for k, v in starmap(lambda k, v: (k, v.squeeze().cpu()), state.items())
        },
    )
