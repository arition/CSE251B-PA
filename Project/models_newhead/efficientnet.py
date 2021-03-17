import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet, get_model_params


class EfficientNetB4Head(EfficientNet):
    def __init__(self, image_size=(256, 256)):
        blocks_args, global_params = get_model_params('efficientnet-b4', None)
        global_params = global_params._replace(image_size=image_size)
        super().__init__(blocks_args, global_params)

        self.start_layer = sum([block_args.num_repeat for block_args in self._blocks_args[:-1]])

    def forward(self, x):
        # Blocks
        for idx, block in enumerate(self._blocks[self.start_layer:]):
            idx = self.start_layer + idx
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


class EfficientNetB4(EfficientNet):
    def __init__(self, image_size=(256, 256)):
        blocks_args, global_params = get_model_params('efficientnet-b4', None)
        global_params = global_params._replace(image_size=image_size)
        super().__init__(blocks_args, global_params)

        self.out_features = self._fc.in_features
        self.image_size = image_size
        self.stop_layer = sum([block_args.num_repeat for block_args in self._blocks_args[:-1]])

    def make_head(self):
        head = nn.Sequential(
            EfficientNetB4Head(self.image_size),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(self._global_params.dropout_rate),
            nn.Flatten()
        )
        return head

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks[:self.stop_layer]):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        return x

    def forward(self, x):
        x = self.extract_features(x)
        return x

    def load_pretrain(self):
        url = 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth'
        pretrained_dict = torch.hub.load_state_dict_from_url(url, progress=False)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
