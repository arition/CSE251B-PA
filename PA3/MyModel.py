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
    def __init__(self,img_ch=3,output_ch=27,t=2):
        super(MyModel,self).__init__()
        
        self.MyMaxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.RRCNN1 = RRCNN_block(inChannel=img_ch,outChannel=32,t=t)
        
        self.RRCNN2 = RRCNN_block(inChannel=32,outChannel=64,t=t)

        self.RRCNN3 = RRCNN_block(inChannel=64,outChannel=128,t=t)
        
        self.RRCNN4 = RRCNN_block(inChannel=128,outChannel=256,t=t)
        
        self.RRCNN5 = RRCNN_block(inChannel=256,outChannel=512,t=t)
        
        self.RRCNN6 = RRCNN_block(inChannel=512,outChannel=1024,t=t)

        self.RRCNN7 = RRCNN_block(inChannel=1024,outChannel=512,t=t)

        self.RRCNN8 = RRCNN_block(inChannel=512,outChannel=256,t=t)

        self.RRCNN9 = RRCNN_block(inChannel=256,outChannel=128,t=t)

        self.RRCNN10 = RRCNN_block(inChannel=128,outChannel=64,t=t)

        self.classifier = RRCNN_block(inChannel=64,outChannel=output_ch,t=t)


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

        x = self.MyMaxpool(x)
        x = self.RRCNN7(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN8(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN9(x)

        x = self.MyMaxpool(x)
        x = self.RRCNN10(x)

        x = self.MyMaxpool(x)
        x = self.classifier(x)

       

        return d1