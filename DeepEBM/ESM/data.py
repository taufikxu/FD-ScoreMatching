import torch
import numpy as np
from torchvision import transforms, datasets
import PIL.Image
import torch.nn.functional as F


def inf_train_gen_imagenet(batch_size, flip=True, train=True, infinity=True):
    if flip:
        transf = transforms.Compose(
            [
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]
        )
    else:
        transf = transforms.ToTensor()
    
    if train is True:
        split = "train"
    else:
        split = "val"

    loader = torch.utils.data.DataLoader(
        datasets.ImageNet("/home/LargeData/ImageNet/", split="train", transform=transf),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )
    if infinity is True:
        while True:
            for img, labels in loader:
                yield img
    else:
        for img, labels in loader:
            yield img


def inf_train_gen_cifar(batch_size, flip=True, train=True, infinity=True):
    if flip:
        transf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]
        )
    else:
        transf = transforms.ToTensor()

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "/home/LargeData/cifar/", train=train, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )
    if infinity is True:
        while True:
            for img, labels in loader:
                # print(img.shape)
                yield img
    else:
        for img, labels in loader:
            yield img


class NumpyImageDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(self.imgs[index])
        # print(img.shape)

        img = img.transpose([1, 2, 0])
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img


def inf_train_gen_celeba(batch_size, flip=True, train=True, infinity=True):
    if flip:
        transf = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
        )
    else:
        transf = transforms.ToTensor()
    imgs = np.load("/home/LargeData/celebA_64x64.npy")
    celeba_dataset = NumpyImageDataset(imgs, transf)

    loader = torch.utils.data.DataLoader(
        celeba_dataset, batch_size, drop_last=True, shuffle=True, num_workers=8
    )
    if infinity is True:
        while True:
            for img in loader:
                yield img
    else:
        for img in loader:
            yield img


def inf_train_gen_celeba32(batch_size, flip=True, train=True, infinity=True):
    if flip:
        transf = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
        )
    else:
        transf = transforms.ToTensor()
    imgs = np.load("/home/LargeData/celebA_64x64.npy")
    celeba_dataset = NumpyImageDataset(imgs, transf)

    loader = torch.utils.data.DataLoader(
        celeba_dataset, batch_size, drop_last=True, shuffle=True, num_workers=8
    )
    if infinity is True:
        while True:
            for img in loader:
                img = img[:, :, ::2, ::2]
                yield img
    else:
        for img in loader:
            img = img[:, :, ::2, ::2]
            yield img


def inf_train_gen_mnist(batch_size, train=True, infinity=True):

    transf = transforms.ToTensor()

    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/home/LargeData/", train=train, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    if infinity is True:
        while True:
            for img, labels in loader:
                img = F.pad(img, [2, 2, 2, 2])
                yield img
    else:
        for img, labels in loader:
            img = F.pad(img, [2, 2, 2, 2])
            yield img


def inf_train_gen_fashionmnist(batch_size, train=True, infinity=True):

    transf = transforms.ToTensor()

    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "/home/LargeData/", train=train, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    if infinity is True:
        while True:
            for img, labels in loader:
                img = F.pad(img, [2, 2, 2, 2])
                yield img
    else:
        for img, labels in loader:
            img = F.pad(img, [2, 2, 2, 2])
            yield img


def inf_train_gen_svhn(batch_size, train=True, infinity=True):
    split = "train" if train is True else "test"

    transf = transforms.ToTensor()

    loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "/home/LargeData/svhn", split=split, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    if infinity is True:
        while True:
            for img, labels in loader:
                yield img
    else:
        for img, labels in loader:
            yield img
