import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FastSegUNet(nn.Module):
    def __init__(self, base_ch: int = 16):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 3, base_ch * 4, base_ch * 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            DSConvBlock(c1, c1, stride=1),
        )
        self.enc2 = nn.Sequential(DSConvBlock(c1, c2, stride=2), DSConvBlock(c2, c2, stride=1))
        self.enc3 = nn.Sequential(DSConvBlock(c2, c3, stride=2), DSConvBlock(c3, c3, stride=1))
        self.enc4 = nn.Sequential(DSConvBlock(c3, c4, stride=2), DSConvBlock(c4, c4, stride=1))
        self.bottleneck = nn.Sequential(DSConvBlock(c4, c5, stride=2), DSConvBlock(c5, c4, stride=1))
        self.dec4 = DSConvBlock(c4 + c4, c3, stride=1)
        self.dec3 = DSConvBlock(c3 + c3, c2, stride=1)
        self.dec2 = DSConvBlock(c2 + c2, c1, stride=1)
        self.dec1 = DSConvBlock(c1 + c1, c1, stride=1)
        self.refine_256 = DSConvBlock(c1, c1, stride=1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x / 255.0
        s1 = self.stem(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        u4 = F.interpolate(b, size=(16, 16), mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([u4, s4], dim=1))
        u3 = F.interpolate(d4, size=(32, 32), mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, s3], dim=1))
        u2 = F.interpolate(d3, size=(64, 64), mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))
        u1 = F.interpolate(d2, size=(128, 128), mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))
        u0 = F.interpolate(d1, size=(256, 256), mode="bilinear", align_corners=False)
        r0 = self.refine_256(u0)
        return self.head(r0)
