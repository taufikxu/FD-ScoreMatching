import os
import logging
import numpy as np
from PIL import Image

import math

from Utils.shortcuts import channel_last
from matplotlib import pyplot as plt

plt.switch_backend("Agg")


def make_grid(tensor,
              nrow=10,
              padding=2,
              normalize=False,
              vrange=None,
              scale_each=False,
              pad_value=0):

    # if list of tensors, convert to a 4D mini-batch Tensor
    nrow = 10 if nrow is None else nrow

    if tensor.ndim == 2:  # single image H x W
        tensor = np.expand_dims(tensor, 0)
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = np.concatenate((tensor, tensor, tensor), 0)
        tensor = np.expand_dims(tensor, 0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = np.copy(tensor)  # avoid modifying tensor in-place

        def norm_ip(img, _min, _max):
            img = np.clip(img, _min, _max)
            img = (img - _min) / (_max - _min + 1e-5)
            return img

        def norm_range(t, _vrange):
            if _vrange is not None:
                return norm_ip(t, _vrange[0], _vrange[1])
            else:
                return norm_ip(t, float(t.min()), float(t.max()))

        tensor = norm_range(tensor, vrange)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = np.zeros((num_channels, height * ymaps + padding,
                     width * xmaps + padding)) + pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:y * height + padding + height -
                 padding, x * width + padding:x * width + padding + width -
                 padding] = tensor[k]
            k = k + 1
    return grid


def save_image(img,
               name,
               shape=10,
               padding=2,
               normalize=True,
               vrange=None,
               scale_each=False,
               pad_value=0):

    from PIL import Image
    grid = make_grid(img,
                     nrow=shape,
                     padding=2,
                     pad_value=0,
                     normalize=normalize,
                     vrange=vrange,
                     scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = (grid * 255).transpose([1, 2, 0]).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(name)


def plot_image(images, name, shape=None, figsize=None, row_names=None):
    images = [np.minimum(np.maximum(img, 0.0), 1.0) for img in images]
    images = channel_last(images, is_array=True)

    len_list = len(images)
    im_list = []
    for i in range(len_list):
        im_list.append(images[i])

    if shape is None:
        unit = int(len_list**0.5)
        shape = (unit, unit)

    imshape = im_list[0].shape
    if imshape[2] == 1:
        im_list = [np.repeat(im, 3, axis=2) for im in im_list]
    else:
        im_list = [im for im in im_list]

    fig_kwargs = dict(nrows=shape[0], ncols=shape[1])
    if figsize is not None:
        fig_kwargs.update({"figsize": figsize})
    fig, axes = plt.subplots(**fig_kwargs)
    for idx, image in enumerate(im_list):
        row = idx // shape[1]
        col = idx % shape[1]
        if col != 0 and row_names is not None:
            axes[row, col].set_ylabel(row_names[row], rotation=0)
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")

    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.tight_layout()
    plt.savefig(name)
    plt.close(fig)


def pile_image(inputs, name, shape=None, path=None, subfolder=True):
    inputs = channel_last(inputs, is_array=True)
    if isinstance(inputs, (list, tuple)):
        list_num = len(inputs)
        len_list = inputs[0].shape[0]
        im_list = []
        for i in range(len_list):
            temp_im = []
            for j in range(list_num):
                temp_im.append(inputs[j][i])
            temp_im = np.concatenate(temp_im, axis=1)
            im_list.append(temp_im)
    else:
        len_list = inputs.shape[0]
        im_list = []
        for i in range(len_list):
            im_list.append(inputs[i])

    imshape = im_list[0].shape
    if imshape[2] == 1:
        im_list = [np.repeat(im, 3, axis=2) for im in im_list]
        imshape = im_list[0].shape[:2]
    else:
        im_list = [im for im in im_list]
        imshape = im_list[0].shape[:2]

    len_list = len(im_list)
    if shape is None:
        unit = int(len_list**0.5)
        shape = (unit, unit)
    size = (shape[0] * imshape[0], shape[1] * imshape[1])
    result = Image.new("RGB", size)
    for i in range(min(len_list, shape[0] * shape[1])):
        x = i // shape[0] * imshape[0]
        y = i % shape[1] * imshape[1]
        temp_im = Image.fromarray(im_list[i])
        result.paste(temp_im, (x, y))

    if path is None:
        logging.critical("Not saving any images, just return.")
        return result

    try:
        result.save(os.path.join(path, name))
    except IOError:
        logging.critical("Unable to Save Images!")
    return result
