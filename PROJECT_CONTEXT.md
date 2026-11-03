# Project Context — Cost-Efficient-Segmentation

## Purpose
Shared handoff notes for AI coding assistants and human collaborators working in this repository. Keep this file factual and update it when project goals, experiments, implementation decisions, or known issues change.

## Workspace
- Project: `Cost-Efficient-Segmentation`
- Git branch: `main`
- Local workspace root: `C:\Users\lokes\OneDrive\Desktop\Cost-Efficient-Segmentation`

## Project Overview
Track B: Cost-Efficient Segmentation under a Compute Budget (CSL7360 Assignment 2).
- Goal: Segment images into main object vs. background under a strict 50 ms single-core CPU time budget and 4 MiB model size limit.
- Dataset: DUTS dataset (DUTS-TR 10,553 images split 90/10 into train/val; DUTS-TE 5,019 images for test evaluation) + 6 captured phone photos.
- Resizing standard: 256 x 256 via `cv2.INTER_AREA`, masks via `cv2.INTER_NEAREST` with threshold >= 128.
- Evaluation metric: Quality $Q = 0.70 \times \text{mIoU} + 0.30 \times \text{BF1}$, and Cost-Efficient Segmentation (CES) score:
  $$\text{CES} = Q \times \left[1 - 0.20 \times \frac{T_{\text{CPU}}}{50\text{ ms}} - 0.10 \times \min\left(1, \frac{S_{\text{disk}}}{4\text{ MiB}}\right)\right]$$

## Hardware Environment
- CPU Model: **AMD Ryzen 5 7520U with Radeon Graphics** (4 cores, 8 logical processors)
- Single-thread enforcement: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `cv2.setNumThreads(1)`, `torch.set_num_threads(1)`

## Final Model (v3 Architecture)
- Architecture: `FastSegUNet` (in `predict.py` and `model.py`)
  - Encoder: Stem with stride 2 + depthwise-separable convolutions (`base_ch=16`), 4 downsampling stages (16 -> 32 -> 48 -> 64 -> bottleneck 96 channels).
  - Decoder: Bilinear upsampling with skip concatenations and depthwise separable blocks (dec4, dec3, dec2, dec1).
  - Refinement: Single depthwise-separable refinement block at full 256 x 256 resolution before the 1x1 conv segmentation head.
  - Optimization: Input normalization (`x / 255.0`) folded directly into `forward()`, optimized via `torch.jit.trace` and `torch.jit.freeze`.
  - Precision: FP16 state dictionary (`weights.pt` / `weights_v3.pt`), stored via `torch.save`.
  - Disk size: **148,255 bytes** (~144.8 KiB, 0.035 of 4 MiB, size penalty: 0.00353).

## Official Submission Results
| Dataset / Split | Number of Images | mIoU | Boundary F1 (BF1) | Quality (Q) | T_CPU (ms) | Model Size ($S_{\text{disk}}$) | CES Score |
|---|---|---|---|---|---|---|---|
| **Validation (DUTS-TR 10%)** | 1,055 | 0.8433 | 0.5995 | 0.7702 | 27.01 ms | 148,255 B (144.8 KiB) | 0.6842 |
| **DUTS-TE (Test)** | 5,019 | 0.6883 | 0.4298 | 0.6107 | 26.04 ms | 148,255 B (144.8 KiB) | **0.5450** |
| **Captured Photos (Overall)** | 6 | 0.8164 | 0.5198 | 0.7274 | 26.04 ms | 148,255 B (144.8 KiB) | 0.6490 |

### Breakdown on Captured Photos
- `photo1` (shoes): mIoU = 0.9313, Obj IoU = 0.9013, BF1 = 0.5487, Q = 0.8165
- `photo2` (shaker bottle): mIoU = 0.9235, Obj IoU = 0.8921, BF1 = 0.5129, Q = 0.8003
- `photo5` (earbuds case): mIoU = 0.9479, Obj IoU = 0.9133, BF1 = 0.6723, Q = 0.8652
- `photo6` (backpack): mIoU = 0.9280, Obj IoU = 0.9240, BF1 = 0.6000, Q = 0.8296
- `photo7` (water tap): mIoU = 0.8553, Obj IoU = 0.7341, BF1 = 0.6346, Q = 0.7891
- `photo8` (potted plant): mIoU = 0.4423, Obj IoU = 0.2845, BF1 = 0.1500, Q = 0.3546

### Failure Cases Analysis (for Report)
1. **`photo8` (potted plant)**: Severe false negatives on the thin, bare twigs/branches against the mottled wall background. The multi-stage strided pooling causes thin linear structures (width < 2 px at 256x256) to vanish in the bottleneck representation, and adjacent background foliage/wall stains triggered false positives.
2. **`photo7` (water tap)**: Mineral/water stain on the wall beneath the tap shared chromatic and intensity characteristics with the shadowed underside of the white/blue tap body, causing the segmentation mask to leak into the vertical background stain.

## Deliverables Generated
- `predict.py`: Configured and verified with `python predict.py --image ... --out ...` (prints `T_CPU_ms` and `S_disk_bytes`).
- `weights.pt`: Updated to v3 weights (148,255 bytes).
- `training_curves_v3.png`: Generated from `train_log_2.csv` (35 epochs).
- `captured/overlays/`: Side-by-side Ground Truth (green) vs Predicted (red) overlays generated for all captured photos.
- `submission_metrics_v3.json`: Machine-readable results and individual breakdown.

## Update Log
- 2026-11-03: Completed official evaluation on DUTS-TR validation split, DUTS-TE test set (5,019 images), and captured photos with v3 files. Generated training curves, overlays, and final metrics report.
