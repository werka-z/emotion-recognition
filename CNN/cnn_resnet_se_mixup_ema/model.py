"""Residual CNN with Squeeze-and-Excitation for FER-2013 emotion recognition.

Input:  1 x 48 x 48 grayscale images
Output: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)

This is the third model in the series, after:
  baseline_CNN/         plain VGG-style Conv-BN-ReLU stack + big Linear head
  cnn_gap_aug_labelsmooth/  same stem + GAP head + stronger aug + label smoothing

It is still a *convolutional* neural network — every learnable layer is a Conv2d.
"ResNet" just means the conv blocks add their input back onto their output
(a skip/shortcut connection, `out = F(x) + x`); there are no recurrent layers,
no transformers, no attention-over-tokens. This is the canonical modern CNN
family (He et al. 2015). The Squeeze-and-Excitation block is also pure CNN
machinery: global-average-pool the feature map to one number per channel, run it
through a tiny two-layer bottleneck, and use the result to rescale each channel.

Why these two upgrades, given where cnn_gap_aug_labelsmooth plateaued (~0.69):

- Residual connections let us go deeper (here an 18-layer-style stack) without the
  optimization trouble a plain deep VGG stack hits. Depth is what buys headroom on
  the confusable FER classes (fear/sad/neutral) once augmentation is maxed out.
- Pre-activation residual blocks (BN-ReLU-Conv order, He et al. 2016) train more
  stably and regularize slightly better than post-activation.
- SE channel attention (~+3% params) lets each block emphasize the feature channels
  that matter for the current image — a reliable small win on FER for almost no cost.
- A GAP head (kept from cnn_gap_aug_labelsmooth) keeps the param count modest.

The training recipe (MixUp/CutMix, EMA weights, longer cosine schedule) lives in
train.py; the architecture here is deliberately the only structural change so the
comparison stays about design choices.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrate channels with a learned per-channel gate.

    Squeeze: global average pool -> one scalar per channel (C numbers).
    Excite:  two FC layers (bottleneck by `reduction`) + sigmoid -> gate in [0,1].
    Scale:   multiply each channel of the feature map by its gate.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)        # squeeze -> (B, C)
        s = self.fc(s).view(b, c, 1, 1)    # excite  -> per-channel gate
        return x * s                       # scale


class PreActResidualSEBlock(nn.Module):
    """Pre-activation residual block (BN-ReLU-Conv x2) with an SE gate on the
    residual branch and a projection shortcut when shape changes.

        out = shortcut(x) + SE(conv(BN-ReLU(conv(BN-ReLU(x)))))
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.se = SEBlock(out_ch, reduction)
        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut only when channels or spatial size change.
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        pre = self.relu(self.bn1(x))           # shared pre-activation
        identity = self.shortcut(pre) if self.shortcut is not None else x
        out = self.conv1(pre)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.se(out)
        return out + identity


class EmotionResNetSE(nn.Module):
    """ResNet-18-style stack adapted to 48x48 grayscale FER input.

    Stem keeps full 48x48 resolution (no early 7x7/stride-2 + maxpool as in
    ImageNet ResNet — the input is already tiny). Four residual stages then
    downsample 48 -> 24 -> 12 -> 6, doubling width each time, followed by GAP.

    width controls the channel schedule (default (64,128,256,512), i.e. ResNet-18
    widths). blocks_per_stage controls depth (default 2 per stage = 18-layer-ish).
    """

    def __init__(self, num_classes: int = 7, width=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2), dropout: float = 0.3,
                 reduction: int = 16):
        super().__init__()
        c0 = width[0]
        # 3x3 stem at full resolution (no downsampling) — input is only 48x48.
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = c0
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(width, blocks_per_stage)):
            for block_idx in range(n_blocks):
                # First block of every stage except the first downsamples (stride 2).
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                stages.append(PreActResidualSEBlock(in_ch, out_ch, stride, reduction))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Final pre-activation (standard for pre-act ResNets) before pooling.
        self.final_bn = nn.BatchNorm2d(in_ch)
        self.final_relu = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_relu(self.final_bn(x))
        x = self.pool(x)
        return self.classifier(x)
