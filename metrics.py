import numpy as np
import torch
import torch.nn.functional as F


class ConfusionAccumulator:
    """Pools one 2x2 confusion matrix over an entire set, then computes mIoU."""

    def __init__(self):
        self.tp_obj = self.fp_obj = self.fn_obj = 0
        self.tp_bg = self.fp_bg = self.fn_bg = 0

    def update(self, pred, gt):
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        self.tp_obj += np.logical_and(pred, gt).sum()
        self.fp_obj += np.logical_and(pred, ~gt).sum()
        self.fn_obj += np.logical_and(~pred, gt).sum()
        pred_bg, gt_bg = ~pred, ~gt
        self.tp_bg += np.logical_and(pred_bg, gt_bg).sum()
        self.fp_bg += np.logical_and(pred_bg, ~gt_bg).sum()
        self.fn_bg += np.logical_and(~pred_bg, gt_bg).sum()

    def miou(self):
        iou_obj = self.tp_obj / max(1, self.tp_obj + self.fp_obj + self.fn_obj)
        iou_bg = self.tp_bg / max(1, self.tp_bg + self.fp_bg + self.fn_bg)
        return 0.5 * (iou_obj + iou_bg)


def _boundary_map(mask_t):
    """mask_t: (N,1,H,W) float 0/1 tensor. Outside-image counts as background."""
    up = F.pad(mask_t, (0, 0, 1, 0))[:, :, :-1, :]
    down = F.pad(mask_t, (0, 0, 0, 1))[:, :, 1:, :]
    left = F.pad(mask_t, (1, 0, 0, 0))[:, :, :, :-1]
    right = F.pad(mask_t, (0, 1, 0, 0))[:, :, :, 1:]
    neighbor_min = torch.min(torch.min(up, down), torch.min(left, right))
    boundary = (mask_t > 0.5) & (neighbor_min < 0.5)
    return boundary.float()


def batch_bf1(pred_batch, gt_batch, window=5):
    """pred_batch, gt_batch: (N,H,W) numpy arrays of 0/1. Returns a list of per-image BF1.

    Uses F.max_pool2d (explicitly allowed) to dilate boundary maps by a window x window
    square instead of cv2.dilate (banned), to vectorize the "within its 5x5 window" check.
    """
    pred_t = torch.from_numpy(pred_batch).unsqueeze(1).float()
    gt_t = torch.from_numpy(gt_batch).unsqueeze(1).float()

    pred_b = _boundary_map(pred_t)
    gt_b = _boundary_map(gt_t)

    pad = window // 2
    gt_b_dilated = (F.max_pool2d(gt_b, kernel_size=window, stride=1, padding=pad) > 0).float()
    pred_b_dilated = (F.max_pool2d(pred_b, kernel_size=window, stride=1, padding=pad) > 0).float()

    n = pred_t.shape[0]
    pred_b_sum = pred_b.view(n, -1).sum(1)
    gt_b_sum = gt_b.view(n, -1).sum(1)
    correct_pred = (pred_b * gt_b_dilated).view(n, -1).sum(1)
    correct_gt = (gt_b * pred_b_dilated).view(n, -1).sum(1)

    scores = []
    for i in range(n):
        if pred_b_sum[i] == 0 and gt_b_sum[i] == 0:
            scores.append(1.0)
            continue
        if pred_b_sum[i] == 0 or gt_b_sum[i] == 0:
            scores.append(0.0)
            continue
        p = (correct_pred[i] / pred_b_sum[i]).item()
        r = (correct_gt[i] / gt_b_sum[i]).item()
        scores.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return scores
