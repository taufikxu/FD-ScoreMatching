import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels=1,
                 size=32,
                 embed_size=256,
                 nfilter=64,
                 nfilter_max=512,
                 **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim
        self.nlabels = nlabels

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers - i), nf_max)
            nf1 = min(nf * 2**(nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y=None):
        if y is not None:
            assert (z.size(0) == y.size(0))
        batch_size = z.size(0)
        if y is None:
            y = torch.zeros(batch_size).type(torch.int64)
            if z.is_cuda:
                y = y.cuda()

        if y.dtype is torch.int64:
            y = y.clamp_(None, self.nlabels - 1)
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self,
                 nlabels=1,
                 size=32,
                 embed_size=256,
                 nfilter=64,
                 nfilter_max=1024):
        super().__init__()
        self.embed_size = embed_size
        self.nlabels = nlabels
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, nlabels)

    def forward(self, x, y=None):
        if y is not None:
            assert (x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(actvn(out))

        index = torch.LongTensor(range(out.size(0)))
        if y is None:
            y = torch.zeros(batch_size).type(torch.int64)
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden,
                                self.fout,
                                3,
                                stride=1,
                                padding=1,
                                bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin,
                                    self.fout,
                                    1,
                                    stride=1,
                                    padding=0,
                                    bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = torch.relu(x)
    return out
