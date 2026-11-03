import os
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

from model import FastSegUNet


@torch.no_grad()
def segment(model, img_256_bgr):
    img_rgb = cv2.cvtColor(img_256_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().mul_(1.0 / 255.0)
    logits = model(x)
    return (logits[0, 0] > 0.0).to(torch.uint8).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duts_te_img", required=True, help="folder of raw DUTS-TE images")
    ap.add_argument("--weights", default="weights.pt")
    ap.add_argument("--base_ch", type=int, default=24)
    args = ap.parse_args()

    model = FastSegUNet(base_ch=args.base_ch)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    traced = torch.jit.trace(model, dummy)
    traced = torch.jit.freeze(traced)
    traced.eval()

    files = sorted(os.listdir(args.duts_te_img))
    if not files:
        raise RuntimeError(f"No images found in {args.duts_te_img}")

    # two warm-up runs, not counted: first run is slow while libraries settle,
    # second lets jit.freeze's extra fusion/optimization kick in
    img0 = cv2.imread(os.path.join(args.duts_te_img, files[0]), cv2.IMREAD_COLOR)
    img0 = cv2.resize(img0, (256, 256), interpolation=cv2.INTER_AREA)
    _ = segment(traced, img0)
    _ = segment(traced, img0)

    total = 0.0
    for f in files:
        img_bgr = cv2.imread(os.path.join(args.duts_te_img, f), cv2.IMREAD_COLOR)
        img_256 = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)  # resize: not timed
        t0 = time.process_time()
        _ = segment(traced, img_256)
        t1 = time.process_time()
        total += (t1 - t0)

    t_cpu_ms = (total / len(files)) * 1000.0
    print(f"Images: {len(files)}")
    print(f"T_CPU_ms (official, DUTS-TE): {t_cpu_ms:.3f}")


if __name__ == "__main__":
    main()
