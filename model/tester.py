import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from utils import getlogger


def _test_convnet(model, num_classes=200, criterion=nn.CrossEntropyLoss().cuda()):
    '''
        Given pytorch model, check model update its
        params correctly
    '''
    test_optimizer = optim.SGD(
        [{"params": model.parameters(), "lr": 0.001}],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )
    test_optimizer.zero_grad()

    # generate test image, label
    test_image = np.ones((1, 3, 224, 224))
    test_image = torch.Tensor(test_image).cuda()
    test_label = torch.ones((1), dtype=torch.long).cuda()

    before = []
    for param in model.parameters():
        before.append(param.detach().clone())

    # process image and loss
    output = model(test_image)
    output = criterion(output, test_label)

    # loss should not be 0
    assert output.item() != 0.
    output.backward()

    test_optimizer.step()
    after = []
    for param in model.parameters():
        after.append(param.detach().clone())

    # variables should change with optim.step()
    for (b_param, a_param) in zip(before, after):
        assert (b_param != a_param).any()


def _test_dataloader(dataloader, normalize=True):
    # checks data fetching
    for image, label in tqdm(dataloader):
        pass


def Test(model, dataloaders):
    logger = getlogger()
    logger.info("Initiating Test...")
    _test_convnet(model)

    logger.info("...Model test complete")
    logger.info("\nValidating loaders...")

    # for dataloader in dataloaders:
    # _test_dataloader(dataloader)
    logger.info("...Done")
