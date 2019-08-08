import torch.nn as nn
import torchvision.models as models

'''
For loading models defined in torchvision.models

model.__dict__ contains ...

    - AlexNet
    - VGG
    - ResNet
    - SqueezeNet
    - DenseNet
    - Inception v3
    - GoogLeNet
    - ShuffleNet v2
    - MobileNet v2
    - ResNeXt

for more information, visit https://pytorch.org/docs/stable/torchvision/models.html
'''


def build_model(
    model_type: str = "alexnet",
    experiment_type: str = "SingleRes",
    lr: float = 0.001,
    lr_decay: int = 20,
    num_classes: int = 200,
    batch_size: int = 32,
    epochs: int = 1000,
    pretrain_path: str = "",
    desc: str = "",
    save: bool = False,
):

    model = models.__dict__[model_type](pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model.cuda()
