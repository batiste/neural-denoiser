import torch
from torch import nn


CHANNELS = 16
DILATIONS = (1, 2, 4, 8, 4, 2)


class WaveformDenoiser(nn.Module):
    def __init__(self, channels=CHANNELS, dilations=DILATIONS):
        super().__init__()
        groups = max(group for group in range(1, min(8, channels) + 1) if channels % group == 0)
        self.input_layer = nn.Conv1d(1, channels, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=9,
                        padding=4 * dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(groups, channels),
                    nn.GELU(),
                )
                for dilation in dilations
            ]
        )
        self.output_layer = nn.Conv1d(channels, 1, kernel_size=9, padding=4)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, noisy):
        features = self.input_layer(noisy)
        for block in self.blocks:
            features = features + block(features)
        predicted_noise = self.output_layer(features)
        return noisy - predicted_noise


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")