"""Gated landmark fusion on top of the cnn_gap_aug_labelsmooth backbone.

Two classes:
  - GatedLandmarkFusion: residual additive fusion of a CNN embedding with an
    encoded landmark vector, controlled by a single learned scalar gate alpha.
  - FusionCNN: the EmotionCNNv2 conv stem + GAP, stripped of its final
    classifier, exposing the 512-d embedding, wired to the fusion head.

NOTE ON EMBEDDING DIM: FUSION_INSTRUCTIONS.md describes a 256-d embedding and a
Linear(256, 7) head. The actual cnn_gap_aug_labelsmooth model
(CNN/cnn_gap_aug_labelsmooth/model.py) ends in a 512-channel conv block ->
GAP -> Dropout -> Linear(512, 7); there is no intermediate Linear(*, 256). The
real embedding is therefore 512-d, so cnn_dim=512 here. The fusion design is
otherwise exactly as specified.

Three fusion modes (for the 4-way ablation, selected by `mode`):
  - "gated"  (default): fused = cnn + tanh(alpha) * mask * lm_feat, alpha init 0.
  - "frozen": same, but alpha pinned at 1.0 and never trained.
  - "concat": no gate; concat [cnn, lm_vec] -> Linear -> ReLU -> Linear.
The CNN-only variant of the ablation is just cnn_gap_aug_labelsmooth itself.
"""

import torch
import torch.nn as nn


class GatedLandmarkFusion(nn.Module):
    """Residual additive fusion with a single scalar gate.

    At init alpha=0 so tanh(0)=0 and the head is a pure CNN classifier; the
    optimizer grows alpha only if the landmark geometry genuinely helps. Same
    principle as LayerScale / Flamingo gating, applied to modality fusion.

    The per-sample `mask` zeroes the landmark branch for images where dlib
    failed, so the model degrades gracefully to pure CNN for those samples with
    no imputation.
    """

    def __init__(self, cnn_dim, lm_dim, hidden=128, num_classes=7, freeze_alpha=False):
        super().__init__()
        self.lm_encoder = nn.Sequential(
            nn.Linear(lm_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, cnn_dim),
        )
        if freeze_alpha:
            # Ablation variant 4: gate pinned open at 1.0, never learned.
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            self.alpha = nn.Parameter(torch.zeros(1))  # THE GATE — starts at 0
        self.classifier = nn.Linear(cnn_dim, num_classes)

    def forward(self, cnn_embed, lm_vec, mask):
        # cnn_embed: (B, cnn_dim)
        # lm_vec:    (B, lm_dim)
        # mask:      (B, 1) — 1.0 where landmarks present, 0.0 where missing
        lm_feat = self.lm_encoder(lm_vec)
        gate = torch.tanh(self.alpha)          # scalar in (-1, 1), starts at 0
        fused = cnn_embed + gate * mask * lm_feat   # residual additive fusion
        return self.classifier(fused)


class ConcatFusion(nn.Module):
    """Ablation variant 2: naive concatenation, no gate.

    Concatenate the raw (masked) landmark vector to the CNN embedding and pass
    through a small MLP head. The mask zeroes missing-landmark inputs just like
    the gated head, so the two differ only in the fusion mechanism.
    """

    def __init__(self, cnn_dim, lm_dim, hidden=None, num_classes=7):
        super().__init__()
        hidden = hidden or cnn_dim
        self.net = nn.Sequential(
            nn.Linear(cnn_dim + lm_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, cnn_embed, lm_vec, mask):
        fused = torch.cat([cnn_embed, mask * lm_vec], dim=1)
        return self.net(fused)


class FusionCNN(nn.Module):
    """EmotionCNNv2 backbone (conv stem + GAP) feeding a fusion head.

    The backbone is a verbatim copy of cnn_gap_aug_labelsmooth/model.py up to —
    but not including — its final Dropout+Linear classifier. It outputs the
    512-d post-GAP embedding. `load_backbone_from` copies the matching weights
    from a trained cnn_gap_aug_labelsmooth checkpoint for the --finetune path.
    """

    EMBED_DIM = 512

    def __init__(self, lm_dim=26, num_classes=7, dropout=0.4, mode="gated"):
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
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        # Identical to EmotionCNNv2.features (48->24->12->6, last block keeps 6).
        self.features = nn.Sequential(
            block(1, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.embed_dropout = nn.Dropout(dropout)  # matches GAP head's pre-classifier dropout

        self.mode = mode
        if mode == "concat":
            self.fusion = ConcatFusion(self.EMBED_DIM, lm_dim, num_classes=num_classes)
        elif mode == "frozen":
            self.fusion = GatedLandmarkFusion(self.EMBED_DIM, lm_dim,
                                              num_classes=num_classes, freeze_alpha=True)
        elif mode == "gated":
            self.fusion = GatedLandmarkFusion(self.EMBED_DIM, lm_dim,
                                              num_classes=num_classes)
        else:
            raise ValueError(f"unknown fusion mode: {mode}")

    def backbone(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.embed_dropout(x)
        return x  # (B, 512)

    def forward(self, x, lm_vec, mask):
        embed = self.backbone(x)
        return self.fusion(embed, lm_vec, mask)

    def load_backbone_from(self, ckpt_path, map_location="cpu"):
        """Copy conv-stem weights from a cnn_gap_aug_labelsmooth checkpoint.

        Only the `features.*` tensors are shared (the GAP/pool layers are
        parameter-free and the classifier differs). Returns how many tensors
        were loaded so the caller can sanity-check.
        """
        ckpt = torch.load(ckpt_path, map_location=map_location)
        src = ckpt["model_state"] if "model_state" in ckpt else ckpt
        own = self.state_dict()
        loaded = {k: v for k, v in src.items()
                  if k.startswith("features.") and k in own and own[k].shape == v.shape}
        own.update(loaded)
        self.load_state_dict(own)
        return len(loaded)

    def backbone_parameters(self):
        return self.features.parameters()

    def head_parameters(self):
        return self.fusion.parameters()

    def set_backbone_requires_grad(self, flag: bool):
        for p in self.features.parameters():
            p.requires_grad = flag
