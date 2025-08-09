import numpy as np

def pixel_accuracy(pred, gt):
    return np.mean(pred == gt)

def mean_iou(pred, gt, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        gt_inds = (gt == cls)
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        union = np.logical_or(pred_inds, gt_inds).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def dice_coefficient(pred, gt, num_classes=2):
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        gt_inds = (gt == cls)
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        dice = (2.0 * intersection) / (pred_inds.sum() + gt_inds.sum() + 1e-10)
        dices.append(dice)
    return np.mean(dices)

def f1_score(pred, gt, num_classes=2):
    f1s = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        gt_inds = (gt == cls)
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        if pred_inds.sum() == 0 or gt_inds.sum() == 0:
            f1 = 0.0
        else:
            precision = intersection / pred_inds.sum()
            recall = intersection / gt_inds.sum()
            if (precision + recall) == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return np.mean(f1s)