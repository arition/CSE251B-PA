import torch
from torch.autograd import backward
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


class ResNet50(ResNet):
    def __init__(self):
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super().__init__(block, layers)
        self.flatten = nn.Flatten()
        self.out_features = self.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def load_pretrain(self):
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        pretrained_dict = torch.hub.load_state_dict_from_url(url, progress=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
