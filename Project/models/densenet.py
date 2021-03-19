import torch
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet


class DenseNet201(DenseNet):
    def __init__(self):
        super().__init__(32, (6, 12, 48, 32), 64)
        self.out_features = self.classifier.in_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def load_pretrain(self):
        url = 'https://download.pytorch.org/models/densenet201-c1103571.pth'
        pretrained_dict = torch.hub.load_state_dict_from_url(url, progress=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class DenseNet121(DenseNet):
    def __init__(self):
        super().__init__(32, (6, 12, 24, 16), 64)
        self.out_features = self.classifier.in_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def load_pretrain(self):
        url = 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
        pretrained_dict = torch.hub.load_state_dict_from_url(url, progress=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
