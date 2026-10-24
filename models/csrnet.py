"""
models/csrnet.py

CSRNet (Li et al., CVPR 2018): VGG-16 front-end (first 10 conv layers,
3 max-pools -> output stride 8) + dilated-convolution back-end that
outputs a 1-channel density map. People count = sum of the density map.

Layer names (frontend / backend / output_layer) match the original
CSRNet-pytorch implementation, so its published checkpoints
(e.g. PartAmodel_best.pth.tar, trained on ShanghaiTech Part A) load directly.
"""

import torch
import torch.nn as nn


FRONTEND_FEAT = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
BACKEND_FEAT = [512, 512, 512, 256, 128, 64]


def _make_layers(cfg, in_channels=3, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3,
                             padding=d_rate, dilation=d_rate)
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, use_vgg_pretrained=False):
        super().__init__()
        self.frontend = _make_layers(FRONTEND_FEAT)
        self.backend = _make_layers(BACKEND_FEAT, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if use_vgg_pretrained:
            # Only useful if you plan to TRAIN from scratch. Needs internet.
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            fs = list(self.frontend.state_dict().items())
            vs = list(vgg.state_dict().items())
            for i in range(len(fs)):
                fs[i][1].data[:] = vs[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def _safe_torch_load(weights_path, device):
    """
    PyTorch >= 2.6 defaults to weights_only=True, which can reject older
    checkpoints (optimizer state, numpy scalars). Try the safe load first
    and fall back only for a file you trust.
    """
    try:
        return torch.load(weights_path, map_location=device)
    except Exception as error:
        print(
            "[CSRNet] Safe load failed "
            f"({type(error).__name__}); retrying with weights_only=False. "
            "Only do this for checkpoints you trust."
        )
        return torch.load(weights_path, map_location=device, weights_only=False)


def load_csrnet(weights_path, device="cpu"):
    """Builds CSRNet and loads a checkpoint (handles the 'state_dict'
    wrapper and 'module.' prefixes from DataParallel).

    Raises RuntimeError if any weights are missing, so the model can never
    silently run with random weights."""
    device = torch.device(device)

    model = CSRNet()
    checkpoint = _safe_torch_load(weights_path, device)

    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
    else:
        state = checkpoint

    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(
            f"CSRNet checkpoint is missing {len(missing)} keys, "
            f"e.g. {missing[:5]}. The checkpoint does not match this "
            "architecture."
        )

    if unexpected:
        print(f"[CSRNet] Note: ignored extra keys: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model