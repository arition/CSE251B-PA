# U-net model
# https://arxiv.org/pdf/1505.04597.pdf
import torch
import torch.nn as nn

# Conv2d operation block, 
class conv_block(nn.Module):
    def __init__(self,inChannel,outChannel):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, outChannel, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,inChannel,outChannel):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inChannel,outChannel,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(outChannel),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self,img_ch=3,n_class=27):
        super(U_Net,self).__init__()
        
        self.UnetMaxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(inChannel=img_ch,outChannel=64)
        self.Conv2 = conv_block(inChannel=64,outChannel=128)
        self.Conv3 = conv_block(inChannel=128,outChannel=256)
        self.Conv4 = conv_block(inChannel=256,outChannel=512)
        self.Conv5 = conv_block(inChannel=512,outChannel=1024)

        self.Up5 = up_conv(inChannel=1024,outChannel=512)
        self.Up_conv5 = conv_block(inChannel=1024, outChannel=512)

        self.Up4 = up_conv(inChannel=512,outChannel=256)
        self.Up_conv4 = conv_block(inChannel=512, outChannel=256)
        
        self.Up3 = up_conv(inChannel=256,outChannel=128)
        self.Up_conv3 = conv_block(inChannel=256, outChannel=128)
        
        self.Up2 = up_conv(inChannel=128,outChannel=64)
        self.Up_conv2 = conv_block(inChannel=128, outChannel=64)

        self.classifier = nn.Conv2d(64,n_class,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x = self.Conv1(x)
        x = self.UnetMaxpool(x)
        x = self.Conv2(x)      
        x = self.UnetMaxpool(x)
        x = self.Conv3(x)
        x = self.UnetMaxpool(x)
        x = self.Conv4(x)
        x = self.UnetMaxpool(x)
        x = self.Conv5(x)

        x = self.Up5(x)
        x = torch.cat((x,x),dim=1)    
        x = self.Up_conv5(x)  
        x = self.Up4(x)
        x = torch.cat((x,x),dim=1)
        x = self.Up_conv4(x)
        x = self.Up3(x)
        x = torch.cat((x,x),dim=1)
        x = self.Up_conv3(x)
        x = self.Up2(x)
        x = torch.cat((x,x),dim=1)
        x = self.Up_conv2(x)
        
        score = self.classifier(x)
        return score