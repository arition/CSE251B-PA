import time

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from U_net import *
from dataloader import *
from utils import *

writer = SummaryWriter()

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = DataLoader(dataset=train_dataset, batch_size=2, num_workers=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=2, num_workers=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=2, num_workers=4, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)


epochs = 20
criterion = nn.CrossEntropyLoss()
fcn_model = U_Net()
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
        for iter, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()
                labels = Y.cuda()
            else:
                inputs, labels = X, Y

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                writer.add_scalar('Loss/train', loss.item(), last_iter)
            last_iter += 1

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'latest_model')

        val(epoch)


def val(epoch):
    print(f'Start val epoch{epoch}')
    iouMetric = IOUMetric()
    pixelAccMetric = PixelAccMetric()
    fcn_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad():
        for iter, (X, Y) in enumerate(val_loader):
            if use_gpu:
                inputs = X.cuda()
                Y = Y.cuda()
            else:
                inputs = X

            ts = time.time()
            outputs = F.log_softmax(fcn_model(inputs), dim=1)
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


if __name__ == "__main__":
    # val(-1)  # show the accuracy before training
    train()
