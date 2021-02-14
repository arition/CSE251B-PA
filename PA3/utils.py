import torch


def iou(pred: torch.Tensor, target: torch.Tensor, n_class: int = 26):
    ious = []
    for cls in range(n_class):
        # Complete this function
        intersection = torch.sum((pred == cls) == (target == cls))
        union = torch.sum(pred == cls) + torch.sum(target == cls) - intersection
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def pixel_acc(pred: torch.Tensor, target: torch.Tensor):
    tp = torch.sum(pred == target)
    return tp / len(pred.flatten())
