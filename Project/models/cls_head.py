import torch.nn as nn


class ClsHead(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.ettHead = nn.Linear(self.backbone.out_features, 3)
        self.ngtHead = nn.Linear(self.backbone.out_features, 4)
        self.cvcHead = nn.Linear(self.backbone.out_features, 3)
        self.sgcHead = nn.Linear(self.backbone.out_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        ett = self.ettHead(x)
        ngt = self.ngtHead(x)
        cvc = self.cvcHead(x)
        sgc = self.sgcHead(x)
        return ett, ngt, cvc, sgc
