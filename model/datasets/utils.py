import torch
import csv
import os
import numpy as np


def _load_weight_from_numpy(model, pretrain_path, fit=True, encoding=None):
    pretrained_weight_numpy = np.load(pretrain_path, encoding=encoding, allow_pickle=True).item()
    pretrained_weight_numpy.pop("fc8", None)
    model_state = model.state_dict()
    layer_weight_iteritems = pretrained_weight_numpy.items()

    conv_list = [0, 4, 8, 10, 12]
    fc_list = [0, 3]
    conv_count = 0
    fc_count = 0
    for layer_name, layer_weight in sorted(layer_weight_iteritems):
        if 'conv' in layer_name:
            model_state["features." + str(conv_list[conv_count]) + ".weight"] = torch.from_numpy(layer_weight[0].transpose(3, 2, 0, 1))
            model_state["features." + str(conv_list[conv_count]) + ".bias"] = torch.from_numpy(layer_weight[1])
            conv_count = conv_count + 1
        elif 'fc' in layer_name:
            model_state["classifier." + str(fc_list[fc_count]) + ".weight"] = torch.from_numpy(layer_weight[0].transpose(1, 0))
            model_state["classifier." + str(fc_list[fc_count]) + ".bias"] = torch.from_numpy(layer_weight[1])
            fc_count = fc_count + 1
    model.load_state_dict(model_state, strict=fit)
    model.cuda()


def _load_weight_from_tf():
    raise NotImplementedError


def _load_weight_from_pt(model, pretrain_path):
    model_state = model.state_dict()
    pretrain_weights = torch.load(pretrain_path)
    conv_list = [0, 4, 8, 10, 12]
    fc_list = [0, 3, 6]
    for k, v in sorted(pretrain_weights.items()):
        if "conv" in k:
            name = "features." + str(conv_list[int(k.split("conv")[1][0]) - 1]) + "." + k.split(".")[-1]
            model_state[name] = v
        elif "fc"in k:
            name = "classifier." + str(fc_list[int(k.split("fc")[1][0]) - 6]) + "." + k.split(".")[-1]
            model_state[name] = v
    model.load_state_dict(model_state, strict=True)
    model.cuda()


def load_weight(model, pretrain_path, fit=True, encoding=None):
    if not os.path.exists(pretrain_path):
        print("GIVEN PATH DOES NOT EXIST! STARTING FROM SCRATCH")
        model.cuda()
        return
    print("Loading weight from {}".format(pretrain_path))
    if pretrain_path.endswith('npy'):
        _load_weight_from_numpy(model, pretrain_path, fit, encoding)

    elif pretrain_path.endswith('pt'):
        _load_weight_from_pt(model, pretrain_path)

    elif pretrain_path.endswith('ckpt'):
        _load_weight_from_tf(model, pretrain_path, fit, encoding)


def _load_csv(csv_file_path):
    with open(csv_file_path) as csv_file:
        csv_file_rows = csv.reader(csv_file, delimiter=",")
        for row in csv_file_rows:
            yield row


def _load_mat(mat_file_path):
    pass


def load_file(file_path):
    if file_path.endswith("csv"):
        return _load_csv(file_path)

    elif file_path.endswith("mat"):
        return _load_mat(file_path)
