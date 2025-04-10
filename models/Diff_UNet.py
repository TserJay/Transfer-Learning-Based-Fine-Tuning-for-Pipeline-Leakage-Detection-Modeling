import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== UNet Block（1D） =====
class UNetBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(UNetBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# ===== Conditional UNet (1D version) =====
class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels=3, condition_channels=3, base_channels=64):
        super(ConditionalUNet1D, self).__init__()
        self.input_conv = UNetBlock1D(in_channels + condition_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            UNetBlock1D(base_channels, base_channels * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            UNetBlock1D(base_channels * 2, base_channels * 4),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            UNetBlock1D(base_channels * 4, base_channels * 2),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            UNetBlock1D(base_channels * 2, base_channels),
        )
        self.output_conv = nn.Conv1d(base_channels, in_channels, kernel_size=1)

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        return self.output_conv(x)
