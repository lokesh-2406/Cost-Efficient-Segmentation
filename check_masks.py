import cv2, os, numpy as np

for f in sorted(os.listdir('captured/masks_256')):
    p = os.path.join('captured/masks_256', f)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    obj_pix = (m >= 128).sum()
    pct = (obj_pix / (256 * 256)) * 100
    print(f"{f}: object pixels={obj_pix} ({pct:.1f}%)")

