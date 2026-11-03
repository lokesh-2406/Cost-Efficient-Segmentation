# Cost-Efficient Segmentation under a Compute Budget (Track B)

CSL7360: Computer Vision — Assignment 2

An ultra-efficient fully convolutional segmentation network (**FastSegUNet**) designed from scratch to segment salient foreground objects from background within an average **50 ms single-core CPU** budget and a **4 MiB** parameter limit.

---

## 1. Final Benchmark Results

Evaluated strictly according to assignment rules:
- **Resizing**: $256 \times 256$ via `cv2.INTER_AREA`, masks via `cv2.INTER_NEAREST` (threshold $\ge 128$).
- **Single-Core CPU Timing**: Enforced single-threaded execution (`OMP/MKL/OPENBLAS/VECLIB = "1"`, `cv2.setNumThreads(1)`, `torch.set_num_threads(1)`), timed using `time.process_time()` across all 5,019 DUTS-TE test images after warm-up.
- **Hardware**: AMD Ryzen 5 7520U with Radeon Graphics.
- **Model Size ($S_{\text{disk}}$)**: **148,255 bytes** ($144.78\text{ KiB}$, stored as FP16 state dict), representing only **0.035 of the 4 MiB limit**.

| Dataset / Split | Images | mIoU | Boundary F1 (BF1) | Quality ($Q$) | $T_{\text{CPU}}$ | Model Size ($S_{\text{disk}}$) | CES Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (DUTS-TR 10%)** | 1,055 | **0.8433** | **0.5995** | **0.7702** | 27.01 ms | 148,255 B (144.8 KiB) | **0.6842** |
| **DUTS-TE (Test — Official)** | 5,019 | **0.6883** | **0.4298** | **0.6107** | **26.04 ms** | 148,255 B (144.8 KiB) | **0.5450** |
| **Captured Photos (Own)** | 6 | **0.8164** | **0.5198** | **0.7274** | 26.04 ms | 148,255 B (144.8 KiB) | **0.6490** |

### Official Score Formulation
$$Q = 0.70 \times \text{mIoU} + 0.30 \times \text{BF1}$$

$$\text{CES} = Q \times \left[1 - 0.20 \times \frac{T_{\text{CPU}}}{50\text{ ms}} - 0.10 \times \min\left(1, \frac{S_{\text{disk}}}{4\text{ MiB}}\right)\right]$$

For DUTS-TE:
- $Q = 0.70 \times 0.6883 + 0.30 \times 0.4298 = 0.6107$
- Time cost: $0.20 \times (26.04 / 50.0) = 0.1042$ (48% speed headroom below budget)
- Size cost: $0.10 \times (148255 / 4194304) = 0.0035$
- **$\text{CES} = 0.6107 \times (1.0 - 0.1042 - 0.0035) = \mathbf{0.5450}$**

---

## 2. Model Architecture: FastSegUNet (v3)

A lightweight U-Net style encoder-decoder built from scratch using Depthwise Separable Convolutions (`DSConvBlock`: $3 \times 3$ depthwise conv + BN + ReLU $\to 1 \times 1$ pointwise conv + BN + ReLU):

- **Stem**: Stride-2 conv ($256 \to 128$) + 1 DSConv block (`base_ch = 16`).
- **Encoder**: 3 stages (`enc2`, `enc3`, `enc4`) with stride 2 downsampling ($128 \to 64 \to 32 \to 16$), channels scaled to $[16, 32, 48, 64]$.
- **Bottleneck**: Stride-2 DSConv to $8 \times 8$, expanding to 96 channels, then returning to 64 channels.
- **Decoder**: 4 stages with bilinear upsampling + skip concatenation from matching encoder stages + DSConv blocks ($8 \to 16 \to 32 \to 64 \to 128$).
- **Refinement**: Single DSConv refinement block operating at full $256 \times 256$ resolution.
- **Head**: $1 \times 1$ convolution mapping to binary segmentation logits.
- **Inference Optimization**: Normalization (`x / 255.0`) is folded into `forward()`, traced with `torch.jit.trace`, and optimized with `torch.jit.freeze` to fuse convolutions, batch normalization, and scaling into single instructions.

---

## 3. Quickstart: Deliverable Inference

To segment any image using the required submission format:

```bash
python predict.py --image captured/images_256/photo1.png --out mask.png
```

Output:
```text
T_CPU_ms: 26.04
S_disk_bytes: 148255
```

---

## 4. Evaluation & Reproduction

### Run Comprehensive Benchmark
To evaluate the validation set (1,055 images), DUTS-TE test set (5,019 images), and captured photos:

```bash
python evaluate_v3_submission.py
```

### Official CPU Timing on DUTS-TE
To measure single-threaded CPU execution over all 5,019 test images:

```bash
python time_official.py --duts_te_img DUTS-TE/DUTS-TE/DUTS-TE-Image --weights weights.pt --base_ch 16
```

### Generate Overlays for Captured Photos
```bash
python generate_overlays_v3.py
```

### Plot Training Progression Curves
```bash
python plot_training_curves.py --log_file train_log_2.csv --out training_curves_v3.png
```

---

## 5. Repository Structure

```text
Cost-Efficient-Segmentation/
├── captured/                       # Captured photos deliverable
│   ├── original/                   # Original taken photos (JPG)
│   ├── images_256/                 # Resized 256x256 test photos (PNG)
│   ├── masks_256/                  # Ground truth binary masks (PNG)
│   └── overlays/                   # True vs predicted semi-transparent overlays
├── model.py                        # FastSegUNet architecture
├── dataset.py                      # DUTS loader & reproducible 90/10 split
├── metrics.py                      # Vectorized mIoU confusion matrix & batch BF1
├── predict.py                      # Official submission CLI inference script
├── weights.pt                      # Final trained FP16 parameters (148 KB)
├── train_log_2.csv                 # 35-epoch training and validation loss/metrics log
├── training_curves_v3.png          # Training curves figure
├── submission_metrics_v3.json      # Complete quantitative benchmark results
├── evaluate_v3_submission.py       # Full benchmark evaluation script
├── time_official.py                # CPU timing benchmark
├── compute_ces.py                  # CES calculation utility
├── generate_overlays_v3.py         # Visual overlay generator
├── CES_TrackB_Kaggle_training_v3.ipynb # Kaggle GPU training notebook
└── README.md
```