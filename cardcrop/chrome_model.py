"""Shared architecture for uploaded ChromeArtifactCNN checkpoints."""
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class ChromeArtifactCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_features: int = 64, num_blocks: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(num_features, in_channels, 3, padding=1)

    def forward(self, x):
        residual = self.tail(self.body(self.head(x)))
        return x - residual


