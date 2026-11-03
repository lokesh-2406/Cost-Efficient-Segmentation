import argparse

import cv2
import numpy as np
import torch

from model import FastSegUNet
from dataset import list_duts_pairs, split_train_val
from metrics import ConfusionAccumulator, batch_bf1


@torch.no_grad()
def predict_mask(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    logits = model(x)
    return (logits[0, 0] > 0.0).to(torch.uint8).numpy()


def load_gt_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)


def evaluate_set(model, pairs, batch_size=32):
    conf = ConfusionAccumulator()
    bf1_scores = []
    preds_buf, gts_buf = [], []

    def flush():
        if preds_buf:
            bf1_scores.extend(batch_bf1(np.stack(preds_buf), np.stack(gts_buf)))
            preds_buf.clear()
            gts_buf.clear()

    for img_path, mask_path in pairs:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        gt = load_gt_mask(mask_path)
        pred = predict_mask(model, img_bgr)

        conf.update(pred, gt)
        preds_buf.append(pred.astype(np.float32))
        gts_buf.append(gt.astype(np.float32))
        if len(preds_buf) >= batch_size:
            flush()
    flush()

    miou = conf.miou()
    bf1 = float(np.mean(bf1_scores)) if bf1_scores else 0.0
    q = 0.70 * miou + 0.30 * bf1
    return {"n": len(pairs), "mIoU": round(float(miou), 4),
            "BF1": round(bf1, 4), "Q": round(float(q), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.pt")
    ap.add_argument("--base_ch", type=int, default=24)
    ap.add_argument("--duts_tr_img", required=True)
    ap.add_argument("--duts_tr_mask", required=True)
    ap.add_argument("--duts_te_img", required=True)
    ap.add_argument("--duts_te_mask", required=True)
    ap.add_argument("--captured_img", default=None, help="captured/images_256")
    ap.add_argument("--captured_mask", default=None, help="captured/masks_256")
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    model = FastSegUNet(base_ch=args.base_ch)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()

    tr_pairs = list_duts_pairs(args.duts_tr_img, args.duts_tr_mask)
    _, val_pairs = split_train_val(tr_pairs, val_frac=args.val_frac, seed=42)
    te_pairs = list_duts_pairs(args.duts_te_img, args.duts_te_mask)

    print("Validation:", evaluate_set(model, val_pairs))
    print("DUTS-TE:   ", evaluate_set(model, te_pairs))

    if args.captured_img and args.captured_mask:
        cap_pairs = list_duts_pairs(args.captured_img, args.captured_mask)
        print("Captured:  ", evaluate_set(model, cap_pairs))


if __name__ == "__main__":
    main()
