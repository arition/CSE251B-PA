import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

def createDeepLabv3(outChannel=27):
    """DeepLabv3 class with custom head
    Args:
        outChannel (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(     
        DeepLabHead(2048, 512),
        #FCN decoding part
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, outChannel, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(outChannel),
        )
    # Set the model in training mode
    model.train()
    return model