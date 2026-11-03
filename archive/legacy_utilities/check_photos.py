import cv2, os, numpy as np

img_dir = 'captured/original'
for f in sorted(os.listdir(img_dir)):
    p = os.path.join(img_dir, f)
    img = cv2.imread(p)
    h, w, c = img.shape
    mean_bgr = img.mean(axis=(0, 1))
    std_bgr = img.std(axis=(0, 1))
    print(f"{f}: shape={w}x{h}, mean_BGR=({mean_bgr[0]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[2]:.1f}), std_BGR=({std_bgr[0]:.1f}, {std_bgr[1]:.1f}, {std_bgr[2]:.1f})")

