"""Basic CNN for FER-2013 facial emotion recognition.

Input:  1 x 48 x 48 grayscale images
Output: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)

A VGG-style stack of conv blocks (Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool)
with channel widths 64 -> 128 -> 256, followed by a small dropout classifier head.
This standard, natural baseline architecture for this dataset.
"""

import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # halves spatial size
            )

        self.features = nn.Sequential(
            block(1, 64),     # 48 -> 24
            block(64, 128),   # 24 -> 12
            block(128, 256),  # 12 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
