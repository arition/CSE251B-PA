################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import caption_utils
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__lr = config_data['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_bleu = 0

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.SGD(self.__model.parameters(), lr=self.__lr, momentum=0.9)
        self.__lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer, factor=0.5, patience=5)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

        self.writer = SummaryWriter()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0

        start_iter = len(self.__train_loader) * self.__current_epoch
        iter_loss = 0

        for i, (images, captions, truths, _) in enumerate(self.__train_loader):
            images = images.to(device)
            captions = captions.to(device)
            truths = truths.to(device)
            self.__optimizer.zero_grad()
            out = self.__model(images, captions)
            loss = self.__criterion(out, truths)
            loss.backward()
            self.__optimizer.step()
            iter_loss += loss.item()
            training_loss += loss.item()

            if i % 10 == 9:
                summary_str = "Epoch: {}, train, Iter: {}, Iter Loss: {}"
                summary_str = summary_str.format(self.__current_epoch + 1, start_iter + i, iter_loss / 10)
                self.__log(summary_str)
                self.writer.add_scalar('train/iter_loss', iter_loss / 10, start_iter + i)
                self.writer.add_scalar('train/learning_rate', self.__optimizer.param_groups[0]['lr'], start_iter + i)
                iter_loss = 0

        training_loss = training_loss / len(self.__train_loader)
        return training_loss

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        start_iter = len(self.__val_loader) * self.__current_epoch
        iter_loss = 0

        with torch.no_grad():
            for i, (images, captions, truths, img_ids) in enumerate(self.__val_loader):
                images = images.to(device)
                captions = captions.to(device)
                truths = truths.to(device)
                out = self.__model(images, captions)
                loss = self.__criterion(out, truths)
                iter_loss += loss.item()
                val_loss += loss.item()

                if i % 10 == 9:
                    summary_str = "Epoch: {}, val, Iter: {}, Iter Loss: {}"
                    summary_str = summary_str.format(self.__current_epoch + 1, start_iter + i, iter_loss / 10)
                    self.__log(summary_str)
                    self.writer.add_scalar('val/iter_loss', iter_loss / 10, start_iter + i)
                    iter_loss = 0

                # only test one mini batch
                if i == 0:
                    val_loss_no_teacher, bleu1, bleu4, text_predicts = self.__test_bleu(
                        images, captions, truths, img_ids, self.__coco)
                    self.writer.add_scalar('val/loss_no_teacher', val_loss_no_teacher /
                                           self.__val_loader.batch_size, self.__current_epoch)
                    self.writer.add_scalar('val/bleu1', bleu1 / self.__val_loader.batch_size, self.__current_epoch)
                    self.writer.add_scalar('val/bleu4', bleu4 / self.__val_loader.batch_size, self.__current_epoch)
                    self.writer.add_text(
                        'val/caption', '\n\n'.join([' '.join(text_predict) for text_predict in text_predicts]), self.__current_epoch)

                    if bleu4 >= self.__best_bleu:
                        self.__best_bleu = bleu4
                        self.__best_model = self.__model.state_dict()
                        self.__save_model(name='best_model.pt')

        val_loss = val_loss / len(self.__val_loader)
        self.__lr_scheduler.step(val_loss)
        return val_loss

    def __test_bleu(self, images, captions, truths, img_ids, coco):
        self.__model.eval()
        bleu1 = 0
        bleu4 = 0
        with torch.no_grad():
            out = self.__model(images, captions)
            loss = self.__criterion(out, truths)
            test_loss = loss

            text_predicts = self.__model.forward_eval(
                images, self.__generation_config['temperature'], self.__generation_config['max_length'])

            for text_predict, img_id in zip(text_predicts, img_ids):
                text_true = []
                for ann in coco.imgToAnns[img_id]:
                    caption = ann['caption']
                    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                    text_true.append(tokens)
                bleu1 += caption_utils.bleu1(text_true, text_predict)
                bleu4 += caption_utils.bleu4(text_true, text_predict)

        return test_loss, bleu1, bleu4, text_predicts

    #  Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.load_state_dict(self.__best_model)
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        text_predicts = []

        with torch.no_grad():
            for iter, (images, captions, truths, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
                truths = truths.to(device)
                test_loss_, bleu1_, bleu4_, text_predicts_ = self.__test_bleu(
                    images, captions, truths, img_ids, self.__coco_test)
                test_loss += test_loss_
                bleu1_ += bleu1_
                bleu4_ += bleu4_
                text_predicts += text_predicts_

        test_loss = test_loss / len(self.__test_loader)
        bleu1 = bleu1 / len(self.__test_loader)
        bleu4 = bleu4 / len(self.__test_loader)

        self.writer.add_scalar('test/loss', test_loss, self.__current_epoch)
        self.writer.add_scalar('test/bleu1', bleu1, self.__current_epoch)
        self.writer.add_scalar('test/bleu4', bleu4, self.__current_epoch)
        self.writer.add_text('test/caption', '\n\n'.join([' '.join(text_predict)
                                                          for text_predict in text_predicts]), self.__current_epoch)

        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1, bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self, name='latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, name)
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.writer.add_scalar('train/loss', train_loss, self.__current_epoch)
        self.writer.add_scalar('val/loss', val_loss, self.__current_epoch)
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
