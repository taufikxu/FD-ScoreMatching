import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
import torchvision
from torchvision import datasets, transforms


from Tools import FLAGS


def get_dataset(train, subset):

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    if FLAGS.dataset.lower() == "svhn":
        if train is True:
            split = "train"
        else:
            split = "test"

        sets = datasets.SVHN(
            "/home/LargeData/Regular/svhn", split=split, download=True, transform=transf
        )
    elif FLAGS.dataset.lower() == "cifar10":
        sets = datasets.CIFAR10(
            "/home/LargeData/Regular/cifar",
            train=train,
            download=True,
            transform=transf,
        )
    elif FLAGS.dataset.lower() == "cifar100":
        sets = datasets.CIFAR100(
            "/home/LargeData/Regular/cifar",
            train=train,
            download=True,
            transform=transf,
        )

    return sets


def inf_train_gen(batch_size, train=True, infinity=True, subset=0):

    loader = torch.utils.data.DataLoader(
        get_dataset(train, subset),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )
    if infinity is True:
        while True:
            for img, labels in loader:
                yield img, labels
    else:
        for img, labels in loader:
            yield img, labels


if __name__ == "__main__":
    # Utils.config.load_config("./configs/classifier_cifar10_mt_aug.yaml")
    FLAGS.zca = True
    FLAGS.translate = 2

    # wrapper = AugmentWrapper()
    dataset = get_dataset(True, 0)
    img_list = []
    for i in range(100):
        img, _ = dataset.__getitem__(i)
        img_list.append(img)
    img_list = torch.stack(img_list, 0).cuda()
    torchvision.utils.save_image((img_list + 1) / 2, "./tmp.png", nrow=10)

    # img_list = wrapper(img_list)
    # print(torch.max(img_list), torch.min(img_list))
    # torchvision.utils.save_image((img_list + 1) / 2, "./tmp1.png", nrow=10)
