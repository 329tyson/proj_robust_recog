import torch

from utils import TrainingConfig
from training.modelwrapper import ModelWrapper


class Baseline(ModelWrapper):
    def __init__(self, config: TrainingConfig):
        super(Baseline, self).__init__(config)

    def fetch_input(self, data):
        if len(data) > 1:
            x, _ = data
        x = torch.squeeze(x)
        return x.cuda().float()

    def fetch_label(self, data):
        _, y = data
        return y.cuda() - 1

    def fetch_loss(self, output, label):
        return self.criterion(output, label)
