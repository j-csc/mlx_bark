# MLX Bark

A port of Suno's [Bark](https://github.com/suno-ai/bark) model in Apple's ML Framework, [MLX](https://github.com/ml-explore/mlx).

Repository is under active development, but the model is functional. 

### Requirements:

Listed in `requirements.txt`

- Python 3.8 or later
- [mlx](https://github.com/ml-explore/mlx)
- [transformers](https://pypi.org/project/transformers/)
- [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [tqdm](https://pypi.org/project/tqdm/)
- [numpy](https://numpy.org/install/)
- [torch](https://pytorch.org/get-started/locally/)
- [encodec](https://pypi.org/project/encodec/)
- [scipy](https://www.scipy.org/install.html)

### Quick Start: 
```bash
# Download weights
huggingface-cli download suno/bark coarse.pt fine.pt text.pt

# Convert to npz format
python convert.py weights/

# Run the model
python model.py weights/
```

### Acknowledgements
Thanks to Suno for the original model, weights and training code repository. Also thanks to the MLX team for the MLX framework and examples.

Links:
- https://github.com/suno-ai/bark
- https://github.com/ml-explore/mlx
