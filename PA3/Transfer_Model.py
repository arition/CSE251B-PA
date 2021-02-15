# Transfe learning
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn




class TransferModel(nn.Module):
    def __init__(self,img_ch=3,outChannel=27):
        super(TransferModel,self).__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
        self.model.train()


        # BCF decoding part, modified the first layer to match the output channels
        self.relu = nn.ReLU(inplace=True)
        self.deconv0 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn0 = nn.BatchNorm2d(1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def foward(self, x):
        x = self.model(x)
        x = self.bn0(self.relu(self.deconv0(x)))
        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))

        score = self.classifier(x)


        return score