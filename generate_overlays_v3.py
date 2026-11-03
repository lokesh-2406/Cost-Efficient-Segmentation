import os
import cv2
import numpy as np
import torch

from predict_v3 import FastSegUNet, load_weights_fp16, segment_photo_in_memory


def overlay(img_bgr, mask_01, color_bgr, alpha=0.45):
    colored = np.zeros_like(img_bgr)
    colored[:] = color_bgr
    mask3 = np.repeat(mask_01[:, :, None], 3, axis=2).astype(bool)
    out = img_bgr.copy()
    out[mask3] = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)[mask3]
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    weights_path = "weights_v3.pt"
    images_dir = "captured/images_256"
    masks_dir = "captured/masks_256"
    out_dirs = ["overlays", "captured/overlays"]

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    model = FastSegUNet(base_ch=16)
    load_weights_fp16(model, weights_path, device="cpu")
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    traced_model = torch.jit.trace(model, dummy)
    traced_model = torch.jit.freeze(traced_model)
    traced_model.eval()

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Generating overlays for {len(files)} captured photos...")

    for fname in files:
        name, _ = os.path.splitext(fname)
        mask_path = os.path.join(masks_dir, name + ".png")
        if not os.path.exists(mask_path):
            print(f"Skipping {fname}: no mask at {mask_path}")
            continue

        img_bgr = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_COLOR)
        if img_bgr.shape[:2] != (256, 256):
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_256 = cv2.resize(gt_raw, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_256 >= 128).astype(np.uint8)

        pred_mask = segment_photo_in_memory(traced_model, img_bgr)

        # True mask (green) | Predicted mask (red)
        true_overlay = label(overlay(img_bgr, gt_mask, (0, 200, 0)), "Ground Truth")
        pred_overlay = label(overlay(img_bgr, pred_mask, (0, 0, 220)), "Predicted (v3)")

        sep = np.ones((256, 3, 3), dtype=np.uint8) * 128
        combined = np.hstack([true_overlay, sep, pred_overlay])

        for d in out_dirs:
            out_file = os.path.join(d, f"{name}_overlay.png")
            cv2.imwrite(out_file, combined)
        print(f"Saved overlay for {name}")

    print("Overlays generated successfully.")


if __name__ == "__main__":
    main()

