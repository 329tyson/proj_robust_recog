import torch.nn as nn
import torchvision.models as models


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

    model = models.__dict__[model_type](pretrained=True, aux_logits=False)
    # model = models.__dict__[model_type](pretrained=False, num_classes=num_classes)

    # pretrained_weights = pretrained_model.state_dict()

    # pop last fc for finetuning
    # pretrained_weights.pop("classifier.6.weight", None)
    # pretrained_weights.pop("classifier.6.bias", None)
    # model.load_state_dict(pretrained_weights, strict=True)

    # last_layer = list(model.modules())[-1]
    model.fc = nn.Linear(2048, num_classes)

    return model.cuda()
