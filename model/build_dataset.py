import os

from torch.utils import data
from datasets.transforms import build_transforms
from datasets.utils import load_file
from datasets.pil_augmentation import get_image
from datasets.pil_augmentation import crop_to_bounding_box
# from datasets.cv2_augmentation import get_image
# from datasets.cv2_augmentation import crop_to_bounding_box


class DatasetWrapper(data.Dataset):
    def __init__(self, imagepath: str, labelpath: str, preprocess):
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.preprocess = preprocess
        self.filepaths = []
        self.labels = []

    def __len__(self):
        return len(self.filepaths)


class CUB200_2011(DatasetWrapper):
    def __init__(
        self,
        imagepath: str,
        labelpath: str,
        preprocess,
        basic_transform,
        is_kd: bool,
        is_test: bool
    ):
        super(CUB200_2011, self).__init__(imagepath, labelpath, preprocess)
        self.bbox = []
        self.is_kd = is_kd
        self.is_test = is_test
        self.basic_transform = basic_transform
        self.read_data_paths()

    def __getitem__(self, idx):
        image = get_image(self.filepaths[idx])
        image = crop_to_bounding_box(image, self.bbox[idx])
        label = self.labels[idx]
        if self.is_kd is True:
            return self.basic_transform(image), self.preprocess(image), label
        else:
            return self.preprocess(image), label

    def read_data_paths(self):
        for row in load_file(self.labelpath):
            self.filepaths.append(os.path.join(self.imagepath, row[0]))
            self.bbox.append((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
            self.labels.append(int(row[5]))


def _build_cub_dataset(
    imagepath: str,
    labelpath: str,
    low_ratio: int,
    mean: list,
    std: list,
    batch_size: int = 128,
    shuffle=True,
    num_workers=6,
    drop_last=True,
    input_shape=(312, 312),
    crop_size=(299, 299),
    is_kd=False,
    is_test=False,
):
    params = {"batch_size": batch_size,
              "shuffle": shuffle,
              "num_workers": num_workers,
              "drop_last": drop_last}

    print("SPECIFIC PREPROCESS")
    preprocess = build_transforms(
        input_shape=input_shape,
        crop_size=crop_size,
        mean=mean,
        std=std,
        low_ratio=low_ratio,
        is_test=is_test,
    )

    if is_kd is True:
        print("BASIC TRANSFORM")
        basic_transform = build_transforms(
            input_shape=input_shape,
            crop_size=crop_size,
            mean=mean,
            std=std,
        )
    else:
        basic_transform = None

    dataset = CUB200_2011(
        imagepath=imagepath,
        labelpath=labelpath,
        preprocess=preprocess,
        basic_transform=basic_transform,
        is_kd=is_kd,
        is_test=is_test,
    )

    return data.DataLoader(dataset, **params)


def build_dataloader(**kwargs):
    if kwargs["dataset_type"].lower() == "cub":
        kwargs.pop("dataset_type", None)
        return _build_cub_dataset(**kwargs)
    else:
        raise NotImplementedError
