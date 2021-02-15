# My model
import torch.nn as nn

class Recurrent_block(nn.Module):
    def __init__(self,outChannel,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.outChannel = outChannel
        self.conv = nn.Sequential(
            nn.Conv2d(outChannel,outChannel,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(outChannel),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x = self.conv(x+x1)
        return x
        
class RRCNN_block(nn.Module):
    def __init__(self,inChannel,outChannel,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(outChannel,t=t),
            Recurrent_block(outChannel,t=t)
        )
        self.Conv_1x = nn.Conv2d(inChannel,outChannel,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x(x)
        x1 = self.RCNN(x)
        return x+x1


class MyModel(nn.Module):
    def __init__(self,img_ch=3,outChannel=27,t=2):
        super(MyModel,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.MyMaxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block(inChannel=img_ch,outChannel=32,t=t)
        
        self.RRCNN2 = RRCNN_block(inChannel=32,outChannel=64,t=t)

        self.RRCNN3 = RRCNN_block(inChannel=64,outChannel=128,t=t)
        
        self.RRCNN4 = RRCNN_block(inChannel=128,outChannel=256,t=t)
        
        self.RRCNN5 = RRCNN_block(inChannel=256,outChannel=512,t=t)
        
        self.RRCNN6 = RRCNN_block(inChannel=512,outChannel=1024,t=t)

        self.deconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32,outChannel,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x = self.RRCNN1(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN2(x)
        
        x = self.MyMaxpool(x)
        x = self.RRCNN3(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN4(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN5(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN6(x)

        x = self.bn1(self.relu(self.deconv1(x)))

        x = self.bn2(self.relu(self.deconv2(x)))

        x = self.bn3(self.relu(self.deconv3(x)))

        x = self.bn4(self.relu(self.deconv4(x)))

        x = self.bn5(self.relu(self.deconv5(x)))

        x = self.bn6(self.relu(self.deconv6(x)))

        x = self.MyMaxpool(x)
        score = self.classifier(x)

       

        return score