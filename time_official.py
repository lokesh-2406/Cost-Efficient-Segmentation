import os
# Set thread limits before importing NumPy, OpenCV or PyTorch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import time

import cv2
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

# Reuse the exact final submission inference implementation. This keeps the
# benchmark's preprocessing, normalization, threshold and model code aligned
# with predict.py and evaluate_v3_submission.py.
from predict import FastSegUNet, load_weights_fp16, resize_photo_256, segment_photo_in_memory


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(
        description="Measure the final single-thread CPU inference path on DUTS-TE."
    )
    ap.add_argument(
        "--duts_te_img",
        required=True,
        help="Folder containing the raw DUTS-TE images.",
    )
    ap.add_argument("--weights", default="weights.pt", help="Final FP16 checkpoint.")
    ap.add_argument(
        "--base_ch",
        type=int,
        default=16,
        help="Model base channel width (the submitted V3 model uses 16).",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.duts_te_img):
        raise FileNotFoundError(f"DUTS-TE image directory not found: {args.duts_te_img}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    files = sorted(
        name
        for name in os.listdir(args.duts_te_img)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        and os.path.isfile(os.path.join(args.duts_te_img, name))
    )
    if not files:
        raise RuntimeError(f"No supported images found in {args.duts_te_img}")

    model = FastSegUNet(base_ch=args.base_ch)
    load_weights_fp16(model, args.weights, device="cpu")

    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    traced_model = torch.jit.trace(model, dummy)
    traced_model = torch.jit.freeze(traced_model)
    traced_model.eval()

    # Two untimed warm-up passes, matching the final V3 evaluation pipeline.
    first_path = os.path.join(args.duts_te_img, files[0])
    first_image = cv2.imread(first_path, cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Could not read DUTS-TE image: {first_path}")
    first_image_256 = resize_photo_256(first_image)
    _ = segment_photo_in_memory(traced_model, first_image_256)
    _ = segment_photo_in_memory(traced_model, first_image_256)

    total_cpu_seconds = 0.0
    for index, filename in enumerate(files, start=1):
        image_path = os.path.join(args.duts_te_img, filename)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read DUTS-TE image: {image_path}")

        # File I/O and resizing are outside the timed region, as specified.
        image_256 = resize_photo_256(image_bgr)

        start = time.process_time()
        _ = segment_photo_in_memory(traced_model, image_256)
        total_cpu_seconds += time.process_time() - start

        if index % 500 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} images")

    t_cpu_ms = (total_cpu_seconds / len(files)) * 1000.0
    print(f"Images: {len(files)}")
    print(f"T_CPU_ms (official, DUTS-TE): {t_cpu_ms:.3f}")


if __name__ == "__main__":
    main()
