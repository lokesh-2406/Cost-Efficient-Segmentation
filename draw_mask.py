"""
Interactive mask-drawing tool for the captured photos.

Loads each 256x256 photo from captured/images_256, lets you paint the object
in white with the mouse, and saves a binary mask (255 = object, 0 = background)
to captured/masks_256 under the same filename.

Controls:
  Left mouse (drag)   paint object (white)
  Right mouse (drag)  erase (back to background / black)
  [ / ]               decrease / increase brush size
  c                   clear current mask
  u                   undo last stroke
  s                   save mask and move to next image
  n                   skip to next image without saving
  p                   go back to previous image
  q / Esc             quit

Usage:
  python draw_mask.py --images_dir captured/images_256 --masks_dir captured/masks_256
"""
import os
import argparse

import cv2
import numpy as np


class MaskDrawer:
    def __init__(self, img_bgr, brush_size=8):
        self.img = img_bgr
        self.mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        self.history = []
        self.brush_size = brush_size
        self.drawing = False
        self.erasing = False
        self.last_pt = None

    def load_existing(self, mask_path):
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == self.mask.shape:
                self.mask = (m >= 128).astype(np.uint8) * 255

    def push_history(self):
        self.history.append(self.mask.copy())
        if len(self.history) > 20:
            self.history.pop(0)

    def undo(self):
        if self.history:
            self.mask = self.history.pop()

    def paint(self, x, y, erase):
        color = 0 if erase else 255
        if self.last_pt is not None:
            cv2.line(self.mask, self.last_pt, (x, y), color, thickness=self.brush_size * 2)
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
        self.last_pt = (x, y)

    def render(self):
        overlay = self.img.copy()
        colored = np.zeros_like(self.img)
        colored[:, :, 2] = self.mask  # red tint over the masked (object) region
        blended = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)
        cv2.putText(blended, f"brush:{self.brush_size}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return blended


def mouse_callback(event, x, y, flags, param):
    drawer = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawer.push_history()
        drawer.drawing = True
        drawer.last_pt = None
        drawer.paint(x, y, erase=False)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawer.push_history()
        drawer.erasing = True
        drawer.last_pt = None
        drawer.paint(x, y, erase=True)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawer.drawing:
            drawer.paint(x, y, erase=False)
        elif drawer.erasing:
            drawer.paint(x, y, erase=True)
    elif event == cv2.EVENT_LBUTTONUP:
        drawer.drawing = False
        drawer.last_pt = None
    elif event == cv2.EVENT_RBUTTONUP:
        drawer.erasing = False
        drawer.last_pt = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="captured/images_256")
    ap.add_argument("--masks_dir", default="captured/masks_256")
    ap.add_argument("--brush_size", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.masks_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    win = "draw_mask  [LMB=paint RMB=erase  [ ]=brush  c=clear u=undo  s=save+next n=skip p=prev  q=quit]"
    cv2.namedWindow(win)

    idx = 0
    while 0 <= idx < len(files):
        fname = files[idx]
        name, _ = os.path.splitext(fname)
        img = cv2.imread(os.path.join(args.images_dir, fname), cv2.IMREAD_COLOR)
        if img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        drawer = MaskDrawer(img, brush_size=args.brush_size)
        mask_path = os.path.join(args.masks_dir, name + ".png")
        drawer.load_existing(mask_path)

        cv2.setMouseCallback(win, mouse_callback, drawer)

        action = None
        while True:
            cv2.imshow(win, drawer.render())
            key = cv2.waitKey(20) & 0xFF

            if key == ord("["):
                drawer.brush_size = max(1, drawer.brush_size - 1)
            elif key == ord("]"):
                drawer.brush_size += 1
            elif key == ord("c"):
                drawer.push_history()
                drawer.mask[:] = 0
            elif key == ord("u"):
                drawer.undo()
            elif key == ord("s"):
                cv2.imwrite(mask_path, drawer.mask)
                print(f"saved {mask_path}  ({idx + 1}/{len(files)})")
                action = "next"
                break
            elif key == ord("n"):
                action = "next"
                break
            elif key == ord("p"):
                action = "prev"
                break
            elif key in (ord("q"), 27):
                action = "quit"
                break

        if action == "quit":
            break
        elif action == "prev":
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
