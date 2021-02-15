import time

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from collections import Counter

from basic_fcn import *
from dataloader import *
from utils import *

import matplotlib.pyplot as plt
from torchvision import transforms
import PIL

writer = SummaryWriter()

train_dataset = IddDataset(csv_file='train.csv', transforms_='rotate')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')

train_loader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, num_workers=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=8, shuffle=False)

weighted = 'False'


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)


nSamples = [2198453584, 519609102, 28929892, 126362539, 58804972, 59032134, 94293190, 2569952, 101519794, 163659019,
            105075839, 47400301, 30998126, 133835645, 135950687, 41339482, 15989829, 104795610, 8062798, 450973,
            94820222, 341805135, 557016609, 71159311, 1465165566, 1823922767, 2775322]
# nSamples = [0.263799707945899,0.0623496126328182,0.00347139330848224,0.0151626584823554,0.00705620284743149,0.00708346076622156,0.0113145513575178,0.000308377030094705,0.0121816954439407,0.0196379863232593,0.0126083970305785,0.00568771869979480,0.00371956753837482,0.0160593811580563,0.0163131720344951,0.00496046100659214,0.00191867240273055,0.0125747714271562,0.000967481766777560,5.41137400203969e-05,0.0113777916682026,0.0410143272724236,0.0668382629702202,0.00853864079553448,0.175810056312377,0.218858517983828,0.000333020054816781]

normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights).cuda()

epochs = 20
if weighted:
    criterion = nn.CrossEntropyLoss(weight=normedWeights)
else:
    criterion = nn.CrossEntropyLoss()

fcn_model = FCN(n_class=n_class)
# fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=0.0001)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()


def train():
    last_iter = 0
    for epoch in range(epochs):
        fcn_model.train()
        ts = time.time()
        epoch_loss = 0
        num_iter = 0
        for iter, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()
                labels = Y.cuda()
            else:
                inputs, labels = X, Y

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            num_iter += 1
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                writer.add_scalar('Loss/train_batch', loss.item(), last_iter)
            last_iter += 1

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'latest_model')
        writer.add_scalar('Loss/train_epoch', epoch_loss / num_iter, epoch)
        val(epoch)


def val(epoch):
    print(f'Start val epoch{epoch}')
    iouMetric = IOUMetric()
    pixelAccMetric = PixelAccMetric()
    fcn_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad():
        epoch_loss = 0
        num_iter = 0
        for iter, (X, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()
                Y = Y.cuda()
            else:
                inputs = X

            ts = time.time()
            outputs = F.log_softmax(fcn_model(inputs), dim=1)
            valoutputs = fcn_model(inputs)
            valloss = criterion(valoutputs, Y)
            epoch_loss += valloss.item()
            num_iter += 1
            _, pred = torch.max(outputs, dim=1)
            iouMetric.update(pred, Y)
            pixelAccMetric.update(pred, Y)
        writer.add_scalar('Loss/Val_epoch', epoch_loss / num_iter, epoch)
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
    fcn_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad():
        for X, Y in test_loader:
            if use_gpu:
                inputs = X.cuda()
                Y = Y.cuda()
            else:
                inputs = X

            outputs = F.log_softmax(fcn_model(inputs), dim=1)
            _, pred = torch.max(outputs, dim=1)
            iouMetric.update(pred, Y)
            pixelAccMetric.update(pred, Y)

    pixel_acc = pixelAccMetric.result()
    iou = iouMetric.result()
    print("Test, pixel acc {}, avg iou {}, time elapsed {}".format(pixel_acc, np.mean(iou), time.time() - ts))
    print(f"ious result: {iou}")


def test_visualization():
    iouMetric = IOUMetric()
    pixelAccMetric = PixelAccMetric()
    fcn_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad():
        for X, Y in test_loader:
            if use_gpu:
                inputs = X.cuda()
                Y = Y.cuda()
            else:
                inputs = X

            outputs = F.log_softmax(fcn_model(inputs), dim=1)
            _, pred = torch.max(outputs, dim=1)
            break
    imgs = []
    for i in pred[0]:
        for j in i:
            imgs.append(labels[j][2])
    imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
    outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
    plt.axis('off')
    plt.imshow(outputimg)
    plt.title('Output Image')
    plt.show()


if __name__ == "__main__":
    # val(-1)  # show the accuracy before training
    train()
    test_visualization()
    writer.close()
