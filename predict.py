
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(1)
torch.set_num_threads(1)


def resize_photo_256(img_bgr):
    if img_bgr.shape[0] == 256 and img_bgr.shape[1] == 256:
        return img_bgr
    return cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)


class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
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
    def __init__(self, base_ch=24):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 3, base_ch * 4, base_ch * 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
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


@torch.no_grad()
def segment_photo_in_memory(model, img_256_bgr):
    img_rgb = cv2.cvtColor(img_256_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().mul_(1.0 / 255.0)
    logits = model(x)
    return (logits[0, 0] > 0.0).to(torch.uint8).numpy()


def load_weights_fp16(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weights.pt")
    parser.add_argument("--base_ch", type=int, default=24)
    args = parser.parse_args()

    s_disk_bytes = os.path.getsize(args.weights)

    model = FastSegUNet(base_ch=args.base_ch)
    load_weights_fp16(model, args.weights, device="cpu")
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    traced_model = torch.jit.trace(model, dummy)
    traced_model.eval()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image from {args.image}")
    img_256 = resize_photo_256(img_bgr)

    mask_01 = segment_photo_in_memory(traced_model, img_256)  # first run: warm-up + output
    cv2.imwrite(args.out, (mask_01 * 255).astype(np.uint8))

    num_runs = 20
    t_start = time.process_time()
    for _ in range(num_runs):
        _ = segment_photo_in_memory(traced_model, img_256)
    t_end = time.process_time()
    avg_cpu_ms = ((t_end - t_start) / num_runs) * 1000.0

    print(f"T_CPU_ms: {avg_cpu_ms:.2f}")
    print(f"S_disk_bytes: {s_disk_bytes}")


if __name__ == "__main__":
    main()
