# CSL7360 A2 — Track B Report Handoff

All numbers below are final and measured (v3 submission). This file is a
reference sheet for writing the report — nothing here needs to be
recomputed. v2 numbers are kept alongside for the "what we tried, what
happened" narrative the Discussion section should tell.

## 1. Method summary (for the report's method section)

**Architecture — FastSegUNet**, a small encoder-decoder FCN (U-Net style) built
entirely from depthwise-separable convolutions (`DSConvBlock`: depthwise 3x3 +
pointwise 1x1, each BN+ReLU), trained from scratch.

- **Stem**: stride-2 conv (256→128) + one DSConv block.
- **Encoder**: 3 stages (`enc2`, `enc3`, `enc4`), each stride-2 DSConv + DSConv,
  taking resolution 128→64→32→16, channels growing `base_ch * [1,2,3,4]`.
- **Bottleneck**: stride-2 DSConv to 8x8, channels `base_ch*6`, back down to `base_ch*4`.
- **Decoder**: 4 stages, each a bilinear upsample + concat with the matching
  encoder skip connection (U-Net style) + DSConv block, taking 8→16→32→64→128.
- **Final submission (v3) uses `base_ch=16`** (bottleneck width 96 channels),
  down from `base_ch=24` (144 channels) in the earlier v2 version.
- **Refinement — moved back to full 256x256, single stage.** v2 ran the
  refine DSConv block at 128x128, right before the final upsample, to save
  time (cut ~24% of total network compute vs. an earlier full-res version).
  v3 reverts that: refine now runs as a single stage *after* the upsample to
  256x256, paid for by the `base_ch` cut above rather than by the refine
  placement itself. **This change did not improve boundary accuracy as
  intended — see §6.** It's kept in the final submission anyway because,
  combined with the narrower channels, the net effect still improved CES.
- **Normalization folded into the model**: `x = x / 255.0` is the first line
  of `forward()`, not a separate Python-side op — lets `torch.jit.freeze`
  fold the constant scale into the stem's first conv weights at freeze time,
  so it costs nothing at inference.
- **Head**: 1x1 conv to 1 output channel (logits), thresholded at 0 for the mask.

**Loss**: hybrid of pixel-weighted BCE + soft Dice. The BCE term up-weights
pixels near ground-truth boundaries (`edge_boost=2.0`) to push the network
harder on edges.

**Inference speed tricks** (all allowed — TorchScript/`torch.compile` are
explicitly permitted):
- `torch.jit.trace` then `torch.jit.freeze` — fuses Conv+BatchNorm+ReLU,
  removes per-op dispatch overhead, and (with normalization folded into the
  model) also fuses away the `/255` scale.
- Two warm-up inference passes before timing starts.
- Single-threaded CPU execution enforced via `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS=1` plus
  `cv2.setNumThreads(1)` / `torch.set_num_threads(1)`, set before any of
  numpy/opencv/torch are imported.
- Weights stored as fp16 on disk, cast to fp32 only after loading.

## 2. Training setup

- **Dataset**: DUTS-TR split 90/10 into train/val, seeded (`seed=42`).
  - **Train: 9,498 images**
  - **Validation: 1,055 images**
- DUTS-TE (5,019 images) never touched during training, tuning, or checkpoint
  selection — only used for the final reported numbers.
- Trained on Kaggle GPU (notebook: `CES_TrackB_Kaggle_training_v3.ipynb`),
  **35 epochs**, Adam, cosine LR schedule, batch size 32, best checkpoint
  selected by validation Q (converged smoothly — val Q 0.526 at epoch 1 to
  0.770 by epoch 35, per `train_log_2.csv`; see `training_curves_v3.png`).
- Weights exported as fp16 (`weights_v3.pt`).

## 3. Timing / hardware

- **CPU**: AMD Ryzen 5 7520U
- Single core, single thread, no subprocesses/multiprocessing (per spec).
- Timed pipeline = resize already in memory → mask in memory only.
- **T_CPU = 26.04 ms/image**, averaged over all 5,019 DUTS-TE images,
  measured with `time.process_time()` (`time_official.py` /
  `predict_v3.py`), one photo at a time, after untimed warm-up runs.
- Budget is 50 ms — **48% headroom**, a large improvement over v2's 15%.

## 4. Model size

- **S_disk = 148,255 bytes (≈0.141 MiB)**, from `os.path.getsize(weights_v3.pt)`.
- Well under the 4 MiB cap — size penalty in CES is negligible either way.

## 5. Results table (put this directly in the report)

| Set | n | mIoU | BF1 | Q |
|---|---|---|---|---|
| Validation | 1,055 | 0.8433 | 0.5995 | 0.7702 |
| DUTS-TE | 5,019 | 0.6883 | 0.4298 | 0.6107 |
| Captured (own photos) | 6 | 0.8164 | 0.5198 | 0.7274 |

**T_CPU = 26.042 ms | S_disk = 148,255 bytes | CES = 0.5450**

CES breakdown:
- Time penalty: 0.20 x (26.042/50) = **0.1042** (10.4% of Q given up to time cost)
- Size penalty: 0.10 x min(1, 148255/4194304) = **0.0035** (0.4% of Q given up to size cost)
- CES = 0.6107 x (1 − 0.1042 − 0.0035) = **0.5450**

## 6. v2 → v3: what changed, and an honest account of what happened

v3 was a deliberate experiment, not a guaranteed win, and it's worth writing
up that way — it's a more convincing report than pretending every change
helped.

**Hypothesis going in**: v2's Q (0.6253 on DUTS-TE) was held back by BF1
(0.4559) specifically, which we attributed to moving the refine block to
128x128 to save time. Moving refine back to full 256x256 resolution, paid
for by shrinking `base_ch` 24→16, should recover boundary accuracy while
staying under budget.

**What actually happened** — the opposite of the targeted effect:

| Metric | v2 | v3 | Change |
|---|---|---|---|
| DUTS-TE mIoU | 0.6980 | 0.6883 | −0.0097 |
| DUTS-TE BF1 | 0.4559 | 0.4298 | **−0.0261 (worse)** |
| DUTS-TE Q | 0.6253 | 0.6107 | −0.0146 (worse) |
| T_CPU | 42.504 ms | 26.042 ms | −38.7% (much faster) |
| S_disk | 249,631 B | 148,255 B | −40.6% (much smaller) |
| **CES** | 0.5153 | **0.5450** | **+0.0297 (better)** |

BF1 got *worse*, not better — the full-resolution refine did not buy back
the boundary accuracy it was meant to. The most likely explanation: cutting
`base_ch` 24→16 reduced the network's total capacity more than the refine
placement change added back, since the pointwise 1x1 convolutions inside
every `DSConvBlock` scale roughly with channels² — a one-third reduction in
channel width cuts a much larger fraction of total parameters/compute than
that ratio suggests. The capacity loss evidently outweighed the benefit of
refining at higher resolution.

**CES still improved** — but for a different reason than intended: the
model simply got much faster and smaller, which cut the time and size
penalties by more than Q dropped (time penalty alone fell from 17.0% to
10.4% of Q). This is worth stating as the actual finding: **for this
architecture, model capacity (channel width) matters more for boundary
quality than refine-stage resolution does** — a useful, specific conclusion
for the Discussion section, and arguably a better one than "we fixed it."

One partial exception: on the 6 captured photos, BF1 *did* improve
(0.4981→0.5198), suggesting the full-res refine may help more on clean,
close-up real photos than on DUTS-TE's broader and harder distribution —
worth a sentence, but n=6 is too small to lean on this heavily.

## 7. Generalization gap — still present, still worth a paragraph

Validation Q (0.7702) is still notably higher than DUTS-TE Q (0.6107) in v3,
same pattern as v2. Candidate explanations:
- The validation split, drawn from the same collection as training data, may
  be systematically easier than DUTS-TE's independently-collected images.
- A `base_ch=16` model has even less capacity than v2's `base_ch=24` — if v2
  already struggled with boundary generalization, a narrower model should be
  expected to struggle somewhat more, which matches what we see.

## 8. Failure cases (pick 2 for the report)

Overlays are in `captured/overlays/*_overlay.png` (green = ground truth, red
= v3 prediction, side by side) — regenerated for v3 by `generate_overlays_v3.py`.

**Primary failure case — `photo8`, a clear and severe one:**
- mIoU **0.442**, BF1 **0.150**, Q **0.354** — by far the worst of the 6.
- Predicted object area (15,089 px) is only ~59% of ground truth (25,507 px)
  — the model is substantially under-segmenting, not just getting edges
  wrong. Open `captured/overlays/photo8_overlay.png` first and describe
  what's visually different about this photo (lighting, clutter, occlusion,
  or object size/distance per the DUTS spec's difficulty conditions) as the
  likely cause.

**Second failure case — pick from the boundary-quality angle:**
Since BF1 is the weak metric everywhere, a good second example is one where
the *region* is mostly right but the *edges* are ragged — check `photo2`
(BF1 0.513, the next-lowest after photo8) or `photo7` (over-predicts object
area: 5,152 px predicted vs. 4,318 px ground truth, despite a comparatively
higher BF1 of 0.635 — worth checking visually whether this is a small/far
object per the spec's conditions, which the model may be over-growing at
the edges). Pick whichever overlay more clearly shows a boundary problem
when you look at it.

## 9. Deliverables checklist

- [x] `captured/original/`, `captured/images_256/`, `captured/masks_256/`
- [x] `captured/overlays/*_overlay.png` (6 images, v3)
- [x] Codebase: `model.py`, `dataset.py`, `metrics.py`, `evaluate.py`,
      `time_official.py`, `prepare_captured.py`, `draw_mask.py`,
      `make_overlays.py` / `generate_overlays_v3.py`, `compute_ces.py`
- [x] `predict_v3.py` + `weights_v3.pt` — **this is the final submission
      pair**; v2's `predict_1.py` / `weights_1.pt` and v1's `predict.py` /
      `weights.pt` are superseded and should be clearly marked or removed
      from the submission to avoid ambiguity about which is final.
- [x] Kaggle training notebook: `CES_TrackB_Kaggle_training_v3.ipynb`
- [x] `submission_metrics_v3.json`, `train_log_2.csv`, `training_curves_v3.png`
      — raw numbers backing every figure in this doc.
- [ ] **Report** — the only thing left. Sections needed: method (§1),
      training setup incl. split sizes and CPU model (§2–3), results table +
      CES (§5), the v2→v3 experiment writeup (§6) — this is strong material
      for the report since it's a genuine, specific finding — generalization
      gap (§7), the 6 overlay figures, and 2 failure cases (§8).

## 10. Before you submit — cleanup flagged

The repo currently has several same-purpose files from different stages
(`predict.py`, `predict_1.py`, `predict_v3.py`; `weights.pt`,
`weights_1.pt`, `weights_v3.pt`, `weights_old_v1.pt`; two `compute_ces`
files; `time_official.py` and `time_official_1.py`) plus a number of loose
root-level images (`mask*.png`, `photo*.jpg`, `download.jpg`) and
exploratory scripts (`check_failure.py`, `check_masks.py`, `check_photos.py`,
`try1.ipynb`) not referenced anywhere in this handoff. Worth a pass to
remove or archive anything that isn't part of the final submission before
zipping/pushing, so the grader isn't left guessing which `predict.py` is real.
