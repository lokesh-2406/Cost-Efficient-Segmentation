import os
import random
from typing import List, Tuple

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_duts_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Pairs each image in image_dir with its matching mask in mask_dir by file stem."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_map = {}
    for fname in os.listdir(mask_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in VALID_EXTS:
            mask_map[stem] = os.path.join(mask_dir, fname)

    pairs = []
    for fname in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in VALID_EXTS and stem in mask_map:
            pairs.append((os.path.join(image_dir, fname), mask_map[stem]))

    return pairs


def split_train_val(
    pairs: List[Tuple[str, str]], val_frac: float = 0.1, seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Splits pairs into (train_pairs, val_pairs) using a fixed random seed."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    val_count = int(len(shuffled) * val_frac)
    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]
    return train_pairs, val_pairs

