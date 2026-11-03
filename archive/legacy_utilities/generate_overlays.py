"""
Generate side-by-side overlay images for captured photos.

For each captured photo, produces a side-by-side image:
  Left:  original photo with ground-truth mask overlay (green)
  Right: original photo with predicted mask overlay (red)

Usage:
  python generate_overlays.py --weights weights_1.pt \
      --images_dir captured/images_256 --masks_dir captured/masks_256 \
      --out_dir overlays
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


def overlay(img_bgr, mask, color_bgr, alpha=0.4):
    """Blend a colored mask onto an image."""
    vis = img_bgr.copy()
    colored = np.zeros_like(vis)
    colored[:] = color_bgr
    region = mask.astype(bool)
    vis[region] = cv2.addWeighted(vis, 1 - alpha, colored, alpha, 0)[region]
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_1.pt")
    ap.add_argument("--base_ch", type=int, default=24)
    ap.add_argument("--images_dir", default="captured/images_256")
    ap.add_argument("--masks_dir", default="captured/masks_256")
    ap.add_argument("--out_dir", default="overlays")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = FastSegUNet(base_ch=args.base_ch)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()

    files = sorted(f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))

    for fname in files:
        stem = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(args.images_dir, fname), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {fname}")
            continue
        if img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # Ground truth mask
        mask_path = os.path.join(args.masks_dir, stem + ".png")
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"skip (no mask): {fname}")
            continue
        gt_mask = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask >= 128).astype(np.uint8)

        # Predicted mask
        pred_mask = predict_mask(model, img)

        # Create side-by-side: GT (green) | Pred (red)
        gt_overlay = overlay(img, gt_mask, (0, 200, 0))    # green
        pred_overlay = overlay(img, pred_mask, (0, 0, 200))  # red

        # Add labels
        cv2.putText(gt_overlay, "Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pred_overlay, "Predicted", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Separator line
        sep = np.ones((256, 3, 3), dtype=np.uint8) * 128

        combined = np.hstack([gt_overlay, sep, pred_overlay])
        out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
        cv2.imwrite(out_path, combined)
        print(f"saved {out_path}")

    print(f"\nAll overlays written to {args.out_dir}/")


if __name__ == "__main__":
    main()

