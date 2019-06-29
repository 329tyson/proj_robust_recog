import argparse
import json
import os
import torch.optim as optim
import torch.nn as nn

from build_model import build_model
from build_dataset import build_dataloader
from training_schemes import single_res_training
from training_schemes import knowledge_distillation_training

__experiment__ = {
    "SingleRes": single_res_training,
    "KDTraining": knowledge_distillation_training,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagepath", type=str)
    parser.add_argument("--train_label", type=str)
    parser.add_argument("--test_label", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--low_ratio", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--experiment_type", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_decay", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mean", type=tuple)
    parser.add_argument("--std", type=tuple)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrain_path", type=str)
    parser.add_argument("--message", type=str, default="no description specified")
    parser.add_argument("--kd", action="store_true")
    parser.add_argument("--save", action="store_true")

    parser.set_defaults(save=False)
    parser.set_defaults(kd=False)
    args = parser.parse_args()

    assert os.environ.get("CONFIG_PATH") is not None

    config = json.load(open(os.environ.get("CONFIG_PATH"), "r"))
    for arg in vars(args):
        if not getattr(args, arg) and arg in config.keys():
            setattr(args, arg, config[arg])
            print("{} : {}".format(arg, getattr(args, arg)))

    print("\nGENERATING MODEL")
    model = build_model(
        model_type=args.model_type,
        experiment_type=args.experiment_type,
        lr=args.lr,
        lr_decay=args.lr_decay,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        pretrain_path=args.pretrain_path,
        message=args.message,
        save=args.save,
    )
    print(model)

    print("GENERATING TRAINLOADER")
    train_loader = build_dataloader(
        imagepath=args.imagepath,
        dataset_type=args.dataset_type,
        labelpath=args.train_label,
        low_ratio=args.low_ratio,
        batch_size=args.batch_size,
        mean=args.mean,
        std=args.std,
        is_kd=args.kd,
        is_test=False,
    )

    print("GENERATING TESTLOADER")
    test_loader = build_dataloader(
        imagepath=args.imagepath,
        dataset_type=args.dataset_type,
        labelpath=args.test_label,
        low_ratio=args.low_ratio,
        batch_size=1,
        mean=args.mean,
        std=args.std,
        is_kd=args.kd,
        is_test=True,
    )

    default_optimizer = optim.SGD(
        [{"params": model.features.parameters()},
         {"params": model.classifier[:6].parameters()},
         {"params": model.classifier[6].parameters(), "lr": args.lr * 10}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # default_optimizer = optim.Adam(
        # [{"params": model.features.parameters()},
         # {"params": model.classifier[:6].parameters()},
         # {"params": model.classifier[6].parameters(), "lr": args.lr * 10}],
        # lr=args.lr,
        # weight_decay=0.0005,
    # )
    print(default_optimizer)

    __experiment__[args.experiment_type](
        epochs=args.epochs,
        model=model,
        optimizer=default_optimizer,
        init_lr=args.lr,
        lr_decay=args.lr_decay,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_function=nn.CrossEntropyLoss(),
    )
