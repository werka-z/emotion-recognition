"""Improved CNN for FER-2013 facial emotion recognition.

Input:  1 x 48 x 48 grayscale images
Output: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)

Changes vs. the baseline (CNN/baseline_CNN/model.py), each aimed at the baseline's
main failure mode — overfitting (train acc 0.74 vs val 0.66) and weak fear/sad:

- One extra conv block (256 -> 512) for a bit more capacity on hard classes.
- Global Average Pooling head instead of the baseline's 256*6*6 -> 256 Linear.
  That single Linear was ~2.4M params and the model's biggest overfitting source;
  GAP removes it almost entirely while keeping (often improving) accuracy.
- A small dropout before the final classifier instead of the heavy 0.5/0.5 head.

The conv stem keeps the same Conv-BN-ReLU x2 + MaxPool pattern as the baseline so
the comparison stays about the design choices, not a totally different network.
"""

import torch.nn as nn


class EmotionCNNv2(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        def block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))  # halves spatial size
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 64),      # 48 -> 24
            block(64, 128),    # 24 -> 12
            block(128, 256),   # 12 -> 6
            block(256, 512, pool=False),  # 6 -> 6, deeper without shrinking further
        )

        # Global Average Pooling -> 512-vector, then a light classifier head.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
