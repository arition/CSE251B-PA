import time

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from dataloader import *
from utils import *


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






writer = SummaryWriter()

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')




train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, num_workers=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=4, shuffle=False)

weighted = 'True'
nSamples = [2198453584,519609102,28929892,126362539,58804972,59032134,94293190,2569952,101519794,163659019,105075839,47400301,30998126,133835645,135950687,41339482,15989829,104795610,8062798,450973,94820222,341805135,557016609,71159311,1465165566,1823922767,2775322]
normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights).cuda()

if weighted:
    criterion = nn.CrossEntropyLoss(weight=normedWeights)
else:
    criterion = nn.CrossEntropyLoss()


epochs = 20
T_model = createDeepLabv3(27)


optimizer = optim.Adam(T_model.parameters(), lr=0.0001)

use_gpu = torch.cuda.is_available()
if use_gpu:
    T_model = T_model.cuda()


def train():
    last_iter = 0
    for epoch in range(epochs):
        T_model.train()
        ts = time.time()
        for iter, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()
                labels = Y.cuda()
            else:
                inputs, labels = X, Y

            outputs = T_model(inputs)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                writer.add_scalar('Loss/train', loss.item(), last_iter)
            last_iter += 1

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(T_model, 'latest_model')

        val(epoch)


def val(epoch):
    print(f'Start val epoch{epoch}')
    iouMetric = IOUMetric()
    pixelAccMetric = PixelAccMetric()
    T_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad():
        for iter, (X, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()
                Y = Y.cuda()
            else:
                inputs = X

            ts = time.time()
            outputs = F.log_softmax(T_model(inputs), dim=1)
            _, pred = torch.max(outputs, dim=1)
            iouMetric.update(pred, Y)
            pixelAccMetric.update(pred, Y)

    pixel_acc = pixelAccMetric.result()
    iou = iouMetric.result()
    print("Val{}, pixel acc {}, avg iou {}, time elapsed {}".format(epoch, pixel_acc, np.mean(iou), time.time() - ts))

    writer.add_scalar('Pixel Accuracy/Val', pixel_acc, epoch)
    writer.add_scalar('IOU/Average/Val', np.mean(iou), epoch)
    writer.add_scalar('IOU/Road/Val', iou[0], epoch)
    writer.add_scalar('IOU/Sidewalk/Val', iou[2], epoch)
    writer.add_scalar('IOU/Car/Val', iou[9], epoch)
    writer.add_scalar('IOU/Billboard/Val', iou[17], epoch)
    writer.add_scalar('IOU/Sky/Val', iou[25], epoch)


def test():
    iouMetric = IOUMetric()
    pixelAccMetric = PixelAccMetric()
    T_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad():
        for X, Y in test_loader:
            if use_gpu:
                inputs = X.cuda()
                Y = Y.cuda()
            else:
                inputs = X

            outputs = F.log_softmax(T_model(inputs), dim=1)
            _, pred = torch.max(outputs, dim=1)
            iouMetric.update(pred, Y)
            pixelAccMetric.update(pred, Y)

    pixel_acc = pixelAccMetric.result()
    iou = iouMetric.result()
    print("Test, pixel acc {}, avg iou {}, time elapsed {}".format(pixel_acc, np.mean(iou), time.time() - ts))
    print(f"ious result: {iou}")


if __name__ == "__main__":
    # val(-1)  # show the accuracy before training
    train()
