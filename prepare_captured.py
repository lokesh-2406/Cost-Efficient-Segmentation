import os
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser(description="Resize captured/original photos to captured/images_256")
    ap.add_argument("--original_dir", default="captured/original")
    ap.add_argument("--out_dir", default="captured/images_256")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for f in sorted(os.listdir(args.original_dir)):
        name, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(os.path.join(args.original_dir, f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {f}")
            continue
        img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(args.out_dir, name + ".png")
        cv2.imwrite(out_path, img_256)
        print(f"{f} -> {out_path}")

    print("\nNow draw masks_256/<name>.png by hand for each file in images_256 "
          "(white=object, black=background), using the same base name.")


if __name__ == "__main__":
    main()