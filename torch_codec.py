from encodec import EncodecModel
import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn.functional as F


def _load_codec_model(device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    return model


# Loads to torch Encodec model
def codec_decode(fine_tokens):
    codec = _load_codec_model("cpu")
    arr = torch.from_numpy(fine_tokens)[None]
    arr = arr.to("cpu")
    arr = arr.transpose(0, 1)
    emb = codec.quantizer.decode(arr)
    out = codec.decoder(emb)
    audio_arr = out.detach().cpu.numpy().squeeze()
    del arr, emb, out
    return audio_arr


if __name__ == "__main__":
    pass
