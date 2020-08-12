import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os


class Trainer():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, train_data_loader, device=None,
                 valid_data_loader=None, start_epoch=0, num_epochs=25, is_debug=False, logger=None, writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule

        self.log_batchs = log_batchs
        self.device = device

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.is_debug = is_debug
        self.cur_epoch = start_epoch

        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.logger = logger
        self.writer = writer

    def fit(self):
        for epoch in range(0, self.start_epoch):
            self.lr_schedule.step()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.append('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            self.logger.append('-' * 60)
            self.cur_epoch = epoch
            self.lr_schedule.step()
            if self.is_debug:
                self._dump_infos()
            self._train()
            self._valid()
            self._save_best_model()
            print()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.device else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best accuracy: %f' % (self.best_acc))
        self.logger.append('best loss: %f' % (self.best_loss))
        self.logger.append('------------------------------------------------------------')

    def _train(self):
        self.model.train()  # Set model to training mode
        losses = []

        for i, (inputs, labels) in enumerate(self.train_data_loader):  # Notice
            if self.device is not None:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze()
            else:
                labels = labels.squeeze()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)  # Notice

            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())  # Notice
            if 0 == i % self.log_batchs or (i == len(self.train_data_loader) - 1):
                local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                batch_mean_loss = np.mean(loss_list)
                print_str = '[%s]\tTraining Batch[%d/%d]\t Class Loss: %.4f\t' \
                            % (local_time_str, i, len(self.train_data_loader) - 1, batch_mean_loss)
                self.logger.append(print_str)
            self.writer.add_scalar('loss/loss_c', batch_mean_loss, self.cur_epoch)

    def _backward(self, loss, loss_list):
        pass

    def _valid(self):
        self.model.eval()
        losses = []
        with torch.no_grad():  # Notice
            for i, (inputs, labels) in enumerate(self.valid_data_loader):
                if self.device:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels = labels.squeeze()
                else:
                    labels = labels.squeeze()

                outputs = self.model(inputs)  # Notice

    def _save_best_model(self):
        pass

    def _get_input(self):
        for i, (inputs, labels) in enumerate(self.valid_data_loader):
            if self.device:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze()
            else:
                labels = labels.squeeze()
