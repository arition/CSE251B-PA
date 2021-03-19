import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
]


class VGG(nn.Module):

    def __init__(self, features, init_weights=True, url='https://download.pytorch.org/models/vgg11-bbd30ac9.pth'):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        if init_weights:
            self._initialize_weights()
        self.out_features = 4096
        self.url = url

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrain(self):
        pretrained_dict = torch.hub.load_state_dict_from_url(self.url, progress=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(pretrained=True, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), url='https://download.pytorch.org/models/vgg11-bbd30ac9.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11'])))
    return model


def vgg11_bn(pretrained=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), url='https://download.pytorch.org/models/vgg11_bn-6002323d.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11_bn'])))
    return model


def vgg13(pretrained=True, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), url='https://download.pytorch.org/models/vgg13-c768596a.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13'])))
    return model


def vgg13_bn(pretrained=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), url='https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13_bn'])))
    return model


def vgg16(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), url='https://download.pytorch.org/models/vgg16-397923af.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16'])))
    return model


def vgg16_bn(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), url='https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16_bn'])))
    return model


def vgg19(pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), url='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg19'])))
    return model


def vgg19_bn(pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), url='https://download.pytorch.org/models/vgg19_bn-c79401a0.pth', **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg19_bn'])))
    return model
