# 🐶 MLX Bark

A port of Suno's [Bark](https://github.com/suno-ai/bark) model in Apple's ML Framework, [MLX](https://github.com/ml-explore/mlx). Bark is a transformer based text-to-audio model that can generate speech and miscellaneous audio i.e. background noise / music.

#### Disclaimer
Repository is under active development, but the model is functional. Currently the model has a few dependencies that are not supported in MLX, such as [encodec](https://github.com/facebookresearch/encodec) and the [tokenizer](https://huggingface.co/bert-base-multilingual-cased). I am working on a port for these dependencies and will update the repository as soon as I have a working solution.

### Example
> Hello World! My name is Bark and I'm running on Apple's new machine learning framework MLX

https://github.com/j-csc/mlx_bark/assets/5698518/3cd6373f-1405-49f2-8c1c-465c768b0ff4

#### TODO
> Sorted by priority
- [ ] Add support for MLX based Encodec
- [ ] Add support for MLX based Tokenizer
- [X] Fix softmax and multinomial sampling issue
- [X] Add support for large model
- [ ] Support for max_gen_duration and history prompts

### Setup

First, install the dependencies:

```bash
pip install -r requirements.txt
```

To convert a model, first download the Bark PyTorch checkpoint and convert
the weights to the MLX format. For example, to convert the `small` model use:

```bash
huggingface-cli download suno/bark coarse.pt fine.pt text.pt
```

Then, convert the weights to the MLX format:

```bash
# for large model, specify --model large instead of small
python convert.py --torch_weights_dir weights/ --model small 
```

### Running the model 
```bash
# Run the model
python model.py --path weights/ --model small --text "hello world my name is bark"
```

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

### Acknowledgements
Thanks to Suno for the original model, weights and training code repository. Also thanks to the MLX team for the MLX framework and examples.

Links:
- https://github.com/suno-ai/bark
- https://github.com/ml-explore/mlx
