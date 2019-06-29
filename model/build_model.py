import torchvision.models as models
# from clfnets.alexnet import build_alexnet
# from clfnets.alexnet_pytorch import build_alexnet

# __clfnets__ = {
    # "alexnet": build_alexnet,
# }


def build_model(
    model_type: str = "alexnet",
    experiment_type: str = "SingleRes",
    lr: float = 0.001,
    lr_decay: int = 20,
    num_classes: int = 200,
    batch_size: int = 32,
    epochs: int = 1000,
    pretrain_path: str = "",
    message: str = "",
    save: bool = False,
):

    # return __clfnets__[model_type](classes=num_classes, pretrained=True)
    pretrained_model =  models.alexnet(pretrained=True)
    model = models.alexnet(pretrained=False, num_classes=200)
    pretrained_weights = pretrained_model.state_dict()
    pretrained_weights.pop("classifier.6.weight", None)
    pretrained_weights.pop("classifier.6.bias", None)
    model.load_state_dict(pretrained_weights, strict=False)
    return model.cuda()
