import torch
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils

from utils import myCustompbar
from utils import getlogger

from tqdm import tqdm
from tensorboardX import SummaryWriter


def decay_lr(optimizer, epoch, lr_decay, init_lr):
    origin_lr = optimizer.param_groups[0]["lr"]
    next_lr = init_lr * (0.5 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = next_lr
    optimizer.param_groups[-1]["lr"] = next_lr * 10
    if origin_lr != next_lr:
        print("lr decayed to {}".format(next_lr))

def single_res_training(
    epochs,
    model,
    optimizer,
    init_lr,
    lr_decay,
    train_loader,
    test_loader,
    loss_function,
):
    logger = getlogger()
    writer = SummaryWriter()
    logger.info("start training")
    for epoch in range(epochs):
        model.train()
        decay_lr(optimizer, epoch, lr_decay, 0.001)

        # pbar = tqdm(
            # enumerate(train_loader),
            # desc="[EPOCH {}]".format(epoch + 1),
            # bar_format="{desc:<5} [B {n_fmt}] [R {rate_fmt}] [loss {postfix[0][loss]}]",
            # postfix=[dict(loss=0.)],
        # )
        pbar = myCustompbar(f"[EPOCH {epoch+1}]", train_loader)
        for i, (x, y) in pbar:
            x_val = x.cuda().float()
            y_val = y.cuda() - 1

            if i == 0:
                x_image = vutils.make_grid(x_val[:1], scale_each=True)
                writer.add_image("train", x_image, i)

            optimizer.zero_grad()
            output = model(x_val)

            loss = loss_function(output, y_val)
            pbar.postfix[0]["loss"] = loss.item()
            loss.backward()
            optimizer.step()

        writer.add_scalar("training_loss", loss.item(), epoch + 1)

        if((epoch + 1) % 10 != 0):
            continue

        model.eval()
        with torch.no_grad():
            hit = 0
            for i, (x, y) in tqdm(enumerate(test_loader)):
                x_val = torch.squeeze(x)
                x_val = x_val.cuda().float()
                y_val = y.cuda() - 1

                if i == 0:
                    x_image = vutils.make_grid(x_val[:1], scale_each=True)
                    writer.add_image("val", x_image, i)
                output = model(x_val)
                output = nn.Softmax(dim=1)(output)
                output = torch.mean(output, dim=0)
                output = output.cpu().detach().numpy()
                if np.argmax(output) == y_val:
                    hit = hit + 1
            logger.info("hit : {}/{}, rate: {}%\n".format(hit, len(test_loader), float(hit / len(test_loader) * 100)))


def knowledge_distillation_training():
    pass
