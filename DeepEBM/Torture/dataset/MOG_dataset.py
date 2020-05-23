import os
# import torch
from torchvision import transforms
from torch.utils.data import Dataset
# import numpy as np
import PIL.Image as Image
import glob


def _transform(img):
    img = transforms.functional.resized_crop(img, 64, 29, 192, 192, 64)
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, [0.5] * 3, [0.5] * 3)
    return img


def _transform_v2(img, size=128):
    # print(img.size)
    img = transforms.functional.resized_crop(img, 64, 29, 192, 192, size)
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, [0.5] * 3, [0.5] * 3)
    return img


class CLEVRDataset(Dataset):
    def __init__(self,
                 root="/home/LargeData/CLEVR_v1.0",
                 phase='train',
                 transform=None,
                 size=None):
        self.root = os.path.join(root, 'images', phase)
        self.all_imgs = glob.glob(os.path.join(self.root, "*.png"))
        if transform is None:
            if size is None:
                self.transform = _transform
            else:
                self.transform = lambda x: _transform_v2(x, size)
        else:
            self.transform = transform

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        img_path = self.all_imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)
        return img_t

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.all_imgs)


if __name__ == "__main__":
    dataset = CLEVRDataset(transform="large")
    A = dataset.__getitem__(0)
    A.save("tmp_ori.png")
