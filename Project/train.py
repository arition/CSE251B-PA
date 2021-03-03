from typing import Sequence

import torch
import torch.cuda.amp
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler
from ignite.engine import Engine, Events
from ignite.handlers import (Checkpoint, DiskSaver, EarlyStopping,
                             global_step_from_engine)
from ignite.metrics import Accuracy, Loss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import XRayDataset
from models import *


def train():
    device = torch.device('cuda')
    dataset = XRayDataset()
    (train_dataset, val_dataset) = random_split(dataset, [len(dataset) - len(dataset) // 9, len(dataset) // 9],
                                                generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, timeout=60)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, timeout=60)

    model = ClsHead(ResNet50()).to(device, non_blocking=True)
    model.backbone.load_pretrain()
    name = 'resnet50'

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()
    criterion_ce = nn.CrossEntropyLoss().to(device, non_blocking=True)
    criterion_bce = nn.BCEWithLogitsLoss().to(device, non_blocking=True)
    writer = SummaryWriter(f'experiments/{name}/tensorboard')

    def train_process(engine: Engine, batch: Sequence[torch.Tensor]):
        model.train()
        optimizer.zero_grad()

        x, ett, ngt, cvc, sgc = batch
        x = x.cuda(non_blocking=True)
        ett = ett.cuda(non_blocking=True)
        ngt = ngt.cuda(non_blocking=True)
        cvc = cvc.cuda(non_blocking=True)
        sgc = sgc.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            ett_pred, ngt_pred, cvc_pred, sgc_pred = model(x)
            ett_loss = criterion_ce(ett_pred, ett)
            ngt_loss = criterion_bce(ngt_pred, ngt)
            cvc_loss = criterion_bce(cvc_pred, cvc)
            sgc_loss = criterion_ce(sgc_pred, sgc)
            loss = ett_loss + ngt_loss + cvc_loss + sgc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss, ett_loss, ngt_loss, cvc_loss, sgc_loss

    def val_process(engine: Engine, batch: Sequence[torch.Tensor]):
        model.eval()

        with torch.no_grad():
            x, ett, ngt, cvc, sgc = batch
            x = x.cuda(non_blocking=True)
            ett = ett.cuda(non_blocking=True)
            ngt = ngt.cuda(non_blocking=True)
            cvc = cvc.cuda(non_blocking=True)
            sgc = sgc.cuda(non_blocking=True)

            ett_pred, ngt_pred, cvc_pred, sgc_pred = model(x)

            return (ett_pred, ngt_pred, cvc_pred, sgc_pred), (ett, ngt, cvc, sgc)

    trainer = Engine(train_process)
    evaluator = Engine(val_process)

    Loss(criterion_ce, output_transform=lambda y_tuple: (y_tuple[0][0], y_tuple[1][0]), device=device).attach(evaluator, 'ett_loss')
    Loss(criterion_bce, output_transform=lambda y_tuple: (y_tuple[0][1], y_tuple[1][1]), device=device).attach(evaluator, 'ngt_loss')
    Loss(criterion_bce, output_transform=lambda y_tuple: (y_tuple[0][2], y_tuple[1][2]), device=device).attach(evaluator, 'cvc_loss')
    Loss(criterion_ce, output_transform=lambda y_tuple: (y_tuple[0][3], y_tuple[1][3]), device=device).attach(evaluator, 'sgc_loss')

    Accuracy(output_transform=lambda y_tuple: (y_tuple[0][0], y_tuple[1][0]), device=device).attach(evaluator, 'ett_accuracy')
    Accuracy(output_transform=lambda y_tuple: (torch.round(torch.sigmoid(y_tuple[0][1])), torch.round(y_tuple[1][1])), is_multilabel=True, device=device).attach(evaluator, 'ngt_accuracy')
    Accuracy(output_transform=lambda y_tuple: (torch.round(torch.sigmoid(y_tuple[0][2])), torch.round(y_tuple[1][2])), is_multilabel=True, device=device).attach(evaluator, 'cvc_accuracy')
    Accuracy(output_transform=lambda y_tuple: (y_tuple[0][3], y_tuple[1][3]), device=device).attach(evaluator, 'sgc_accuracy')

    to_save = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'trainer': trainer, 'scaler': scaler}
    handler = Checkpoint(to_save, DiskSaver(f'experiments/{name}/checkpoint', create_dir=True), n_saved=2)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    def score_function(engine: Engine):
        return (engine.state.metrics['ett_accuracy'] + engine.state.metrics['ngt_accuracy'] +
                engine.state.metrics['cvc_accuracy'] + engine.state.metrics['sgc_accuracy'])
    handler = Checkpoint(
        {'model': model}, DiskSaver(f'experiments/{name}/checkpoint', create_dir=True),
        n_saved=2, filename_prefix='best', score_function=score_function, score_name="val_acc",
        global_step_transform=global_step_from_engine(trainer)
    )
    evaluator.add_event_handler(Events.COMPLETED, handler)

    handler = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Iter[{trainer.state.iteration}] Loss: {trainer.state.output[0]:.2f}")

        writer.add_scalar('train/loss', trainer.state.output[0], trainer.state.iteration)
        writer.add_scalar('train/ett_loss', trainer.state.output[1], trainer.state.iteration)
        writer.add_scalar('train/ngt_loss', trainer.state.output[2], trainer.state.iteration)
        writer.add_scalar('train/cvc_loss', trainer.state.output[3], trainer.state.iteration)
        writer.add_scalar('train/sgc_loss', trainer.state.output[4], trainer.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        loss = metrics['ett_loss'] + metrics['ngt_loss'] + metrics['cvc_loss'] + metrics['sgc_loss']

        writer.add_scalar('val/loss', loss, trainer.state.epoch)
        writer.add_scalar('val/ett_loss', metrics['ett_loss'], trainer.state.epoch)
        writer.add_scalar('val/ngt_loss', metrics['ngt_loss'], trainer.state.epoch)
        writer.add_scalar('val/cvc_loss', metrics['cvc_loss'], trainer.state.epoch)
        writer.add_scalar('val/sgc_loss', metrics['sgc_loss'], trainer.state.epoch)

        writer.add_scalar('val/ett_accuracy', metrics['ett_accuracy'], trainer.state.epoch)
        writer.add_scalar('val/ngt_accuracy', metrics['ngt_accuracy'], trainer.state.epoch)
        writer.add_scalar('val/cvc_accuracy', metrics['cvc_accuracy'], trainer.state.epoch)
        writer.add_scalar('val/sgc_accuracy', metrics['sgc_accuracy'], trainer.state.epoch)

        lr_scheduler.step(loss)

    trainer.run(train_dataloader, 10000)


if __name__ == "__main__":
    train()
