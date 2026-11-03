import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

cv2.setNumThreads(1)
torch.set_num_threads(1)

from predict_v3 import FastSegUNet, load_weights_fp16, segment_photo_in_memory
from dataset import list_duts_pairs, split_train_val
from metrics import ConfusionAccumulator, batch_bf1


def compute_ces(q_score: float, t_cpu_ms: float, s_disk_bytes: int) -> float:
    if t_cpu_ms > 50.0:
        return 0.0
    four_mib = 4 * 1024 * 1024
    time_penalty = 0.20 * (t_cpu_ms / 50.0)
    size_penalty = 0.10 * min(1.0, s_disk_bytes / four_mib)
    return float(q_score * (1.0 - time_penalty - size_penalty))


def load_gt_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask from {path}")
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)


def evaluate_pairs(model, pairs, batch_size=32, track_timing=False):
    conf = ConfusionAccumulator()
    bf1_scores = []
    preds_buf, gts_buf = [], []
    cpu_times = []

    def flush():
        if preds_buf:
            bf1_scores.extend(batch_bf1(np.stack(preds_buf), np.stack(gts_buf)))
            preds_buf.clear()
            gts_buf.clear()

    # warm-up runs if timing
    if track_timing and len(pairs) > 0:
        img0 = cv2.imread(pairs[0][0], cv2.IMREAD_COLOR)
        img0_256 = cv2.resize(img0, (256, 256), interpolation=cv2.INTER_AREA)
        _ = segment_photo_in_memory(model, img0_256)
        _ = segment_photo_in_memory(model, img0_256)

    total_pairs = len(pairs)
    print(f"Evaluating {total_pairs} pairs (track_timing={track_timing})...")
    start_wall = time.time()

    for idx, (img_path, mask_path) in enumerate(pairs):
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_256 = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)  # resize not timed
        gt = load_gt_mask(mask_path)

        if track_timing:
            t0 = time.process_time()
            pred = segment_photo_in_memory(model, img_256)
            t1 = time.process_time()
            cpu_times.append(t1 - t0)
        else:
            pred = segment_photo_in_memory(model, img_256)

        conf.update(pred, gt)
        preds_buf.append(pred.astype(np.float32))
        gts_buf.append(gt.astype(np.float32))

        if len(preds_buf) >= batch_size:
            flush()

        if (idx + 1) % 500 == 0 or (idx + 1) == total_pairs:
            elapsed = time.time() - start_wall
            print(f"  [{idx + 1}/{total_pairs}] done in {elapsed:.1f}s")

    flush()

    miou = conf.miou()
    bf1 = float(np.mean(bf1_scores)) if bf1_scores else 0.0
    q = 0.70 * miou + 0.30 * bf1
    t_cpu_ms = (np.sum(cpu_times) / len(cpu_times) * 1000.0) if cpu_times else 0.0

    return {
        "n": len(pairs),
        "mIoU": float(miou),
        "BF1": float(bf1),
        "Q": float(q),
        "T_CPU_ms": float(t_cpu_ms),
    }


def main():
    weights_path = "weights_v3.pt"
    s_disk_bytes = os.path.getsize(weights_path)
    print(f"Weights file: {weights_path} ({s_disk_bytes} bytes, {s_disk_bytes/1024:.2f} KiB)")

    # Build and trace model
    model = FastSegUNet(base_ch=16)
    load_weights_fp16(model, weights_path, device="cpu")
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    traced_model = torch.jit.trace(model, dummy)
    traced_model = torch.jit.freeze(traced_model)
    traced_model.eval()

    # 1. Validation Set (10% of DUTS-TR)
    duts_tr_img = "DUTS-TR/DUTS-TR/DUTS-TR-Image"
    duts_tr_mask = "DUTS-TR/DUTS-TR/DUTS-TR-Mask"
    tr_pairs = list_duts_pairs(duts_tr_img, duts_tr_mask)
    train_pairs, val_pairs = split_train_val(tr_pairs, val_frac=0.10, seed=42)
    print(f"\n--- 1. Validation Set ({len(val_pairs)} images) ---")
    val_res = evaluate_pairs(traced_model, val_pairs, batch_size=32, track_timing=True)
    val_ces = compute_ces(val_res["Q"], val_res["T_CPU_ms"], s_disk_bytes)
    print(f"Validation: mIoU={val_res['mIoU']:.4f}, BF1={val_res['BF1']:.4f}, Q={val_res['Q']:.4f}, T_CPU={val_res['T_CPU_ms']:.2f} ms, CES={val_ces:.4f}")

    # 2. DUTS-TE Test Set (5,019 images)
    duts_te_img = "DUTS-TE/DUTS-TE/DUTS-TE-Image"
    duts_te_mask = "DUTS-TE/DUTS-TE/DUTS-TE-Mask"
    te_pairs = list_duts_pairs(duts_te_img, duts_te_mask)
    print(f"\n--- 2. DUTS-TE Test Set ({len(te_pairs)} images, official timing) ---")
    te_res = evaluate_pairs(traced_model, te_pairs, batch_size=32, track_timing=True)
    te_ces = compute_ces(te_res["Q"], te_res["T_CPU_ms"], s_disk_bytes)
    print(f"DUTS-TE: mIoU={te_res['mIoU']:.4f}, BF1={te_res['BF1']:.4f}, Q={te_res['Q']:.4f}, T_CPU={te_res['T_CPU_ms']:.2f} ms, CES={te_ces:.4f}")

    # 3. Captured Photos
    captured_img_dir = "captured/images_256"
    captured_mask_dir = "captured/masks_256"
    cap_pairs = list_duts_pairs(captured_img_dir, captured_mask_dir)
    print(f"\n--- 3. Captured Photos ({len(cap_pairs)} images) ---")
    cap_res = evaluate_pairs(traced_model, cap_pairs, batch_size=1, track_timing=True)
    cap_ces = compute_ces(cap_res["Q"], cap_res["T_CPU_ms"], s_disk_bytes)
    print(f"Captured Overall: mIoU={cap_res['mIoU']:.4f}, BF1={cap_res['BF1']:.4f}, Q={cap_res['Q']:.4f}, T_CPU={cap_res['T_CPU_ms']:.2f} ms, CES={cap_ces:.4f}")

    # 4. Individual Captured Photos evaluation
    print("\n--- 4. Individual Captured Photos Breakdown ---")
    indiv_results = []
    for img_path, mask_path in cap_pairs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        gt = load_gt_mask(mask_path)
        img_256 = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        t0 = time.process_time()
        pred = segment_photo_in_memory(traced_model, img_256)
        t1 = time.process_time()
        c_time = (t1 - t0) * 1000.0

        c_accum = ConfusionAccumulator()
        c_accum.update(pred, gt)
        miou_i = c_accum.miou()
        bf1_i = batch_bf1(pred[None, ...].astype(np.float32), gt[None, ...].astype(np.float32))[0]
        q_i = 0.70 * miou_i + 0.30 * bf1_i
        ces_i = compute_ces(q_i, c_time, s_disk_bytes)

        tp = np.logical_and(pred == 1, gt == 1).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()
        iou_obj = tp / max(1, tp + fp + fn)

        indiv_results.append({
            "name": stem,
            "mIoU": miou_i,
            "IoU_obj": iou_obj,
            "BF1": bf1_i,
            "Q": q_i,
            "T_CPU_ms": c_time,
            "CES": ces_i,
            "gt_obj_pixels": int(gt.sum()),
            "pred_obj_pixels": int(pred.sum()),
        })
        print(f"  {stem:8s}: mIoU={miou_i:.4f} (Obj IoU={iou_obj:.4f}), BF1={bf1_i:.4f}, Q={q_i:.4f}, T_CPU={c_time:.2f} ms, CES={ces_i:.4f}")

    # Print markdown table
    print("\n" + "=" * 80)
    print("SUBMISSION TABLE (LaTeX / Markdown format)")
    print("=" * 80)
    print("| Split / Dataset | mIoU | Boundary F1 | Quality (Q) | T_CPU (ms) | S_disk | CES Score |")
    print("|---|---|---|---|---|---|---|")
    print(f"| **Validation (DUTS-TR 10%)** | {val_res['mIoU']:.4f} | {val_res['BF1']:.4f} | {val_res['Q']:.4f} | {val_res['T_CPU_ms']:.2f} ms | {s_disk_bytes} B ({s_disk_bytes/1024:.1f} KiB) | {val_ces:.4f} |")
    print(f"| **DUTS-TE (Test)** | {te_res['mIoU']:.4f} | {te_res['BF1']:.4f} | {te_res['Q']:.4f} | {te_res['T_CPU_ms']:.2f} ms | {s_disk_bytes} B ({s_disk_bytes/1024:.1f} KiB) | **{te_ces:.4f}** |")
    print(f"| **Captured Photos** | {cap_res['mIoU']:.4f} | {cap_res['BF1']:.4f} | {cap_res['Q']:.4f} | {cap_res['T_CPU_ms']:.2f} ms | {s_disk_bytes} B ({s_disk_bytes/1024:.1f} KiB) | {cap_ces:.4f} |")
    print("=" * 80)

    # Save results to a file for report generation
    import json
    summary_data = {
        "weights_file": weights_path,
        "s_disk_bytes": s_disk_bytes,
        "validation": {**val_res, "CES": val_ces},
        "duts_te": {**te_res, "CES": te_ces},
        "captured": {**cap_res, "CES": cap_ces},
        "captured_individual": indiv_results,
    }
    with open("submission_metrics_v3.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("\nWrote summary metrics to submission_metrics_v3.json")


if __name__ == "__main__":
    main()

