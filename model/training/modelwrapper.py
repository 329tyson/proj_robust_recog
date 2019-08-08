import time
import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from utils import TrainingConfig
from utils import AverageMeter
from utils import getlogger
from utils import config_pbar

class ModelWrapper:
    def __init__(self, config: TrainingConfig):
        self.model = config.model
        self.epochs = config.epochs
        self.optimizer = config.optimizer
        self.init_lr = config.init_lr
        self.train_loader = config.train_loader
        self.valid_loader = config.valid_loader
        self.criterion = config.criterion.cuda()
        self.logger = getlogger()

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)

        self.iterate()

    def train(self, epoch):
        monitor = AverageMeter()
        self.model.train()
        pbar = config_pbar(self._display_header(epoch), self.train_loader)

        for data in pbar:
            x_input = self.fetch_input(data)
            y_label = self.fetch_label(data)

            self.optimizer.zero_grad()
            output = self.model(x_input)
            loss = self.fetch_loss(output, y_label)
            monitor.update(loss.item())

            if isinstance(pbar, tqdm):
                pbar.postfix[0]["loss"] = monitor.avg
                pbar.postfix[0]["live"] = loss.item()

            loss.backward()
            self.optimizer.step()
        # log some output
        self.logger.info(f"[TRAIN {epoch+1} loss : {monitor.avg:.3f}]")

    def validate(self, epoch):
        monitor = AverageMeter()
        total = 0
        self.model.eval()
        with torch.no_grad():
            pbar = config_pbar(self._display_header(epoch), self.valid_loader)

            for data in pbar:
                x_input = self.fetch_input(data)
                y_label = self.fetch_label(data)

                output = self.model(x_input)
                loss, hit = self.ten_crop_eval(output, y_label)
                if hit:
                    total += 1

                monitor.update(loss.item())

                if isinstance(pbar, tqdm):
                    pbar.postfix[0]["loss"] = monitor.avg
                    pbar.postfix[0]["live"] = loss.item()
            # self.lr_scheduler.step(monitor.avg)
            # log some output
            self.logger.info(f"[VALID {epoch+1} loss : {monitor.avg:3.f}]")
            self.logger.info(f"[HIT: {total}/{len(self.valid_loader)} ACC: {total/len(self.valid_loader)*100:.2f}]\n")

    def iterate(self):
        start = time.time()
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
        end = time.time()
        time_spent = end - start
        self.logger.info(f"[Time {time_spent:.3f}({time_spent / self.epochs:.3f} for epoch)")

    def _display_header(self, epoch):
        return f"[EPOCH {epoch + 1}]"

    def ten_crop_eval(self, output, label):
        '''
        Attribues:
            output: predictions on 10 images(10, 200)
            label: answer label for 10 crop images

            this function calculates on following procedure
            1. Doing softmax on dim=1
            2. Averaging across dim=0
            3. Calculate on validation loss
        '''
        output = nn.Softmax(dim=1)(output)
        output = torch.mean(output, dim=0, keepdim=True)
        hit = True if torch.argmax(output, dim=1) == label else False
        return self.criterion(output, label), hit

    def fetch_input(self, data):
        raise NotImplementedError

    def fetch_label(self, data):
        raise NotImplementedError

    def fetch_loss(self, output, label):
        raise NotImplementedError
