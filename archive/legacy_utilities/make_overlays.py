"""
Generates side-by-side (true mask overlay | predicted mask overlay) images for
every captured photo, for the report's required figure.

Usage:
  python make_overlays.py --images_dir captured/images_256 --masks_dir captured/masks_256 \
      --weights weights.pt --out_dir captured/overlays
"""
import os
import argparse

import cv2
import numpy as np
import torch

from model import FastSegUNet


@torch.no_grad()
def predict_mask(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    logits = model(x)
    return (logits[0, 0] > 0.0).to(torch.uint8).numpy()


def load_gt_mask(path, shape):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)


def overlay(img_bgr, mask_01, color_bgr, alpha=0.45):
    colored = np.zeros_like(img_bgr)
    colored[:] = color_bgr
    mask3 = np.repeat(mask_01[:, :, None], 3, axis=2).astype(bool)
    out = img_bgr.copy()
    out[mask3] = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)[mask3]
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="captured/images_256")
    ap.add_argument("--masks_dir", default="captured/masks_256")
    ap.add_argument("--weights", default="weights.pt")
    ap.add_argument("--base_ch", type=int, default=24)
    ap.add_argument("--out_dir", default="captured/overlays")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = FastSegUNet(base_ch=args.base_ch)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()

    files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    for fname in files:
        name, _ = os.path.splitext(fname)
        mask_path = os.path.join(args.masks_dir, name + ".png")
        if not os.path.exists(mask_path):
            print(f"skip {fname}: no ground-truth mask found at {mask_path}")
            continue

        img_bgr = cv2.imread(os.path.join(args.images_dir, fname), cv2.IMREAD_COLOR)
        if img_bgr.shape[:2] != (256, 256):
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        gt_mask = load_gt_mask(mask_path, img_bgr.shape[:2])
        pred_mask = predict_mask(model, img_bgr)

        true_overlay = label(overlay(img_bgr, gt_mask, (0, 200, 0)), "True mask")       # green
        pred_overlay = label(overlay(img_bgr, pred_mask, (0, 0, 220)), "Predicted mask")  # red

        combined = cv2.hconcat([true_overlay, pred_overlay])
        out_path = os.path.join(args.out_dir, name + "_overlay.png")
        cv2.imwrite(out_path, combined)
        print(f"saved {out_path}")

    print(f"\nDone. {len(files)} overlays written to {args.out_dir}")


if __name__ == "__main__":
    main()
