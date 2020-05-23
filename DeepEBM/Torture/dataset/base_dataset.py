import os
import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image


class InfinityDataloaderWraper(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._dataiterator = iter(self.dataloader)

    def __next__(self):
        try:
            batch = self._dataiterator.next()
        except StopIteration:
            self._dataiterator = iter(self.dataloader)
            batch = self._dataiterator.next()
        return batch

    def next(self):
        return self.__next__()


class NumpyImageLabelDataset(Dataset):
    def __init__(self, imgs, label, transform=None, require_onehot=False):
        self.imgs = imgs
        self.label = label
        self.transform = transform

        self.num_classes = np.unique(label).shape[0]
        self.require_onehot = require_onehot

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(self.imgs[index])
        label = self.label[index]
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, axis=2)

        img = (img * 255.0).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class NumpyImageDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(self.imgs[index])
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, axis=2)

        img = (img * 255.0).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img


class ImageInFolder(Dataset):
    def __init__(self, root, ext=".jpg", transform=None):
        self.imgs = glob.glob(root + "/*" + ext)
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.transform = transform

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.pil_loader(self.imgs[index])

        if self.transform is not None:
            img = self.transform(img)

        return img


if __name__ == "__main__":
    # default setting for Caltech101
    tfs = transforms.Compose(
        transforms.Resize(256), transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    print(tfs)
