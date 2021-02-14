import torch


class IOUMetric():
    def __init__(self, n_class: int = 26) -> None:
        self.n_class = n_class
        self.ious = [[0, 0] for cls in range(n_class)]

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for cls in range(self.n_class):
            # Complete this function
            intersection = torch.sum((pred == cls) & (target == cls))
            union = torch.sum(pred == cls) + torch.sum(target == cls) - intersection

            self.ious[cls][0] += intersection.item()
            self.ious[cls][1] += union.item()

    def result(self):
        ious = []
        for cls in range(self.n_class):
            if self.ious[cls][1] == 0:
                # if there is no ground truth, do not include in evaluation
                ious.append(float('nan'))
            else:
                ious.append(self.ious[cls][0] / self.ious[cls][1])

        return ious


class PixelAccMetric():
    def __init__(self, n_class: int = 26) -> None:
        self.n_class = n_class
        self.tp = 0
        self.num = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for cls in range(self.n_class):
            self.tp += torch.sum((pred == cls) & (target == cls)).item()
            self.num += torch.sum(target == cls).item()

    def result(self):
        return self.tp / self.num
