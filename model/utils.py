import os
import logging
import subprocess

import numpy as np

from tqdm import tqdm
from typing import NamedTuple


class TrainingConfig(NamedTuple):
    '''
    Attributes:
        model: network model to train
        epochs: number of epochs to train
        optimizer: optimizer to use in train
        init_lr: initial learning rate
        train_loader: dataloader for training
        valid_loader: dataloader for test(validation)
        criterion: loss function for calculate loss
    '''
    model: object
    epochs: int
    optimizer: object
    init_lr: float
    train_loader: object
    valid_loader: object
    criterion: object


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _streamHandler():
    return logging.StreamHandler()


def _fileHandler(filename):
    return logging.FileHandler(filename)


def _set_visible_devices(num=0):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num)


def getlogger():
    return logging.getLogger(__name__)


def config_logger(filename, test=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    filename = os.path.join(os.getcwd(), filename)
    if test:
        # only print to stdout when testing
        logger.addHandler(_streamHandler())
        return logger

    fileHandler = _fileHandler(filename)
    fileHandler.setLevel(logging.INFO)

    logger.addHandler(fileHandler)

    return logger


def config_pbar(description, loader, background=True):
    if background:
        return loader

    return tqdm(
        loader,
        desc=description,
        bar_format="{desc:<5} [B {n_fmt}] [R {rate_fmt}] [loss {postfix[0][loss]:.3f}] ({postfix[0][live]:.3f})",
        postfix=[dict(loss=0., live=0.)],
    )


def available_gpu():
    gpu_info = subprocess.run(
        args="nvidia-smi | grep Default | cut -d '|' -f 3",
        shell=True,
        stdout=subprocess.PIPE
    )
    gpu_info = gpu_info.stdout.decode("ascii")
    gpu_mem = [x.split()[2] for x in gpu_info.split("\n")[:-1]]

    device_num = np.argmax(np.array(gpu_mem))
    _set_visible_devices(device_num)
    return device_num
