import torch
from efficientnet_pytorch import EfficientNet, get_model_params


class EfficientNetB4(EfficientNet):
    def __init__(self):
        blocks_args, global_params = get_model_params('efficientnet-b4', None)
        global_params = global_params._replace(image_size=(256, 256))
        super().__init__(blocks_args, global_params)
        self.out_features = self._fc.in_features

    def forward(self, x):
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        return x

    def load_pretrain(self):
        url = 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth'
        pretrained_dict = torch.hub.load_state_dict_from_url(url, progress=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
