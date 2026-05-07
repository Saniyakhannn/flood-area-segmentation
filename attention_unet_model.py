# attention_unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Unchanged from original — two conv-BN-ReLU blocks."""
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.3))
        layers += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────
# NEW: Attention Gate
# ─────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Attention Gate as in Oktay et al. 2018.

    Args:
        F_g  : channels in the gating signal (from decoder)
        F_l  : channels in the skip connection (from encoder)
        F_int: intermediate channel size (typically F_l // 2)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # 1x1 conv to project gating signal to F_int channels
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # 1x1 conv to project skip features to F_int channels
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # 1x1 conv to collapse F_int → 1 channel attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()   # output α ∈ [0, 1]
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g : gating signal  — upsampled decoder feature  [B, F_g, H, W]
        x : skip connection — encoder feature            [B, F_l, H, W]
        """
        g1 = self.W_g(g)   # [B, F_int, H, W]
        x1 = self.W_x(x)   # [B, F_int, H, W]

        # Add (broadcast if spatial sizes differ slightly)
        psi = self.relu(g1 + x1)   # [B, F_int, H, W]
        psi = self.psi(psi)        # [B, 1, H, W]  ← attention map α

        # Multiply attention map across all channels of x
        return x * psi             # [B, F_l, H, W]  ← attended features


# ─────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Encoder (identical to original) ──────────────────
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512, use_dropout=True)

        # ── Decoder ──────────────────────────────────────────
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # CHANGE: AttentionGate before concat at level 3
        # F_g=256 (from up1), F_l=256 (skip x3), F_int=128
        self.att1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv1 = DoubleConv(512, 256)   # 256+256 channels in

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # CHANGE: AttentionGate before concat at level 2
        # F_g=128 (from up2), F_l=128 (skip x2), F_int=64
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv2 = DoubleConv(256, 128)   # 128+128 channels in

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # CHANGE: AttentionGate before concat at level 1
        # F_g=64 (from up3), F_l=64 (skip x1), F_int=32
        self.att3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv3 = DoubleConv(128, 64)    # 64+64 channels in

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)               # [B,  64, 256, 256]
        x2 = self.down2(self.pool1(x1))  # [B, 128, 128, 128]
        x3 = self.down3(self.pool2(x2))  # [B, 256,  64,  64]
        x4 = self.bottleneck(self.pool3(x3))  # [B, 512, 32, 32]

        # Decoder with attention gates
        # Level 3
        g = self.up1(x4)                 # [B, 256, 64, 64]
        x3_att = self.att1(g=g, x=x3)   # CHANGE: gate filters skip x3
        x = torch.cat([g, x3_att], dim=1)  # [B, 512, 64, 64]
        x = self.conv1(x)                # [B, 256, 64, 64]

        # Level 2
        g = self.up2(x)                  # [B, 128, 128, 128]
        x2_att = self.att2(g=g, x=x2)   # CHANGE: gate filters skip x2
        x = torch.cat([g, x2_att], dim=1)  # [B, 256, 128, 128]
        x = self.conv2(x)                # [B, 128, 128, 128]

        # Level 1
        g = self.up3(x)                  # [B, 64, 256, 256]
        x1_att = self.att3(g=g, x=x1)   # CHANGE: gate filters skip x1
        x = torch.cat([g, x1_att], dim=1)  # [B, 128, 256, 256]
        x = self.conv3(x)                # [B, 64, 256, 256]

        return self.final(x)             # [B, 1, 256, 256]