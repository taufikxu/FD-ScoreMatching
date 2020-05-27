import torch
import torch.nn.functional as F
from torch import nn

from Utils.flags import FLAGS

# from torch.autograd import grad


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.2)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)


class NormalizedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()

    def forward(self, x):
        y = self.sp(x)
        log2 = torch.log(torch.ones_like(x) * 2)
        y = y - log2
        return y


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, bias=True):
        super(MyConvo2d, self).__init__()

        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input):
        output = self.conv(input)
        # print(input.shape, output.shape)
        return output


class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()
        pass

    def forward(self, in_vect):
        return in_vect ** 2


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, in_vect):
        return in_vect * nn.functional.sigmoid(in_vect)


class ConvSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        return torch.sum(x_in, dim=(1, 2, 3))


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)
        self.pool = nn.AvgPool2d(2, stride=2, padding=0)

    def forward(self, input):
        output = input
        output = self.pool(output)
        output = self.conv(output)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)
        self.pool = nn.AvgPool2d(2, stride=2, padding=0)

    def forward(self, input):
        output = self.conv(input)
        output = self.pool(output)
        return output


class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )

        self.embed = nn.Embedding(num_classes, num_features * 3)
        self.embed.weight.data[:, : 2 * num_features].normal_(1, 0.02)
        self.embed.weight.data[:, 2 * num_features :].zero_()

    def forward(self, x):
        y = torch.zeros(x.shape[0]).to(x.device).type(torch.long)
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
        h = h + means[..., None, None] * alpha[..., None, None]
        out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(
            -1, self.num_features, 1, 1
        )
        return out


class MLP_Quadratic(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(MLP_Quadratic, self).__init__()

        latent_dim = 5000
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.AF = AF
        self.fc1 = nn.Linear(hw * hw * inchan, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, latent_dim)

        self.ln1 = nn.Linear(latent_dim, 1)
        self.ln2 = nn.Linear(latent_dim, 1)
        self.lq = nn.Linear(latent_dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        x_in = x_in.view(-1, self.inchan * self.hw * self.hw)
        output = self.AF(self.fc1(x_in))
        output = self.AF(self.fc2(output))
        output = self.AF(self.fc3(output))
        output = self.AF(self.fc4(output))
        output = self.AF(self.fc5(output))

        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        hw,
        resample=None,
        normalize=False,
        AF=nn.ELU(),
    ):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.normalize = normalize
        self.bn1 = None
        self.bn2 = None
        self.relu1 = AF
        self.relu2 = AF
        if resample == "down":
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == "none":
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])

        if normalize == "instance":
            self.normalize = True
            self.bn1 = InstanceNorm2dPlus(input_dim)
            self.bn2 = InstanceNorm2dPlus(input_dim)

        # print(self.normalize, self.bn1, self.bn2)

        if resample == "down":
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1)
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False
            )
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)
        elif resample == "none":
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size=1)
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False
            )
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size=kernel_size)

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample is None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        if self.normalize is False:
            output = input
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.relu2(output)
            output = self.conv_2(output)
        else:
            output = input
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.conv_2(output)

        return shortcut + output


class Res18_Quadratic(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res18_Quadratic, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.lq = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in.view(-1, self.inchan, self.hw, self.hw)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = output.view(-1, int(self.hw / 8) * int(self.hw / 8) * 8 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        # print(output.shape)
        # exit()
        return output


def reshape_factor(x_in, factor, in_c, hw):
    x_out = x_in.reshape(-1, in_c, hw // factor, factor, hw // factor, factor)
    x_out = x_out.permute(0, 1, 3, 5, 2, 4)
    x_out = x_out.reshape(-1, in_c * factor * factor, hw // factor, hw // factor)
    return x_out


class Res18_Quadratic_dense(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res18_Quadratic_dense, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim + self.inchan,
            2 * dim,
            3,
            int(hw),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc1 = ResidualBlock(
            2 * dim + self.inchan * 4,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim + self.inchan * 4,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim + self.inchan * 4,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim + self.inchan * 16,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim + self.inchan * 16,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim + self.inchan * 16,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim + self.inchan * 64,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim + self.inchan * 64,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim + 32 * 32 * 3, 1)
        self.ln2 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim + 32 * 32 * 3, 1)
        self.lq = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim + 32 * 32 * 3, 1)
        self.Square = Square()

    def forward(self, x_in):
        x_in = x_in.view(-1, self.inchan, self.hw, self.hw)
        x_2 = reshape_factor(x_in, 2, self.inchan, self.hw)
        x_4 = reshape_factor(x_in, 4, self.inchan, self.hw)
        x_8 = reshape_factor(x_in, 8, self.inchan, self.hw)
        x_flatten = torch.zeros_like(x_in).view(-1, self.inchan * self.hw * self.hw)

        output = self.conv1(x_in)
        output = self.rb1(torch.cat([output, x_in], 1))
        output = self.rbc1(torch.cat([output, x_2], 1))
        output = self.rbc11(torch.cat([output, x_2], 1))
        output = self.rb2(torch.cat([output, x_2], 1))
        output = self.rbc2(torch.cat([output, x_4], 1))
        output = self.rbc22(torch.cat([output, x_4], 1))
        output = self.rb3(torch.cat([output, x_4], 1))
        output = self.rbc3(torch.cat([output, x_8], 1))
        output = self.rbc33(torch.cat([output, x_8], 1))
        output = output.view(-1, int(self.hw / 8) * int(self.hw / 8) * 8 * self.dim)
        output = torch.cat([output, x_flatten], 1)

        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class Res18_Quadratic_unet(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res18_Quadratic_unet, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.final_fc1 = nn.Sequential(
            nn.Conv2d(dim, 1, 3, stride=1, padding=1, bias=False), ConvSum()
        )
        self.final_fc2 = nn.Sequential(
            nn.Conv2d(2 * dim, 1, 3, stride=1, padding=1, bias=False), ConvSum(),
        )

        self.final_fc3 = nn.Sequential(
            nn.Conv2d(4 * dim, 1, 3, stride=1, padding=1, bias=False), ConvSum(),
        )

        self.final_fc4 = nn.Sequential(
            nn.Conv2d(8 * dim, 1, 3, stride=1, padding=1, bias=False), ConvSum(),
        )

        # self.ln1 = nn.Linear(
        #     int(hw / 8) * int(hw / 8) * 8 * dim + 32 * 32 * 3, 1)
        # self.ln2 = nn.Linear(
        #     int(hw / 8) * int(hw / 8) * 8 * dim + 32 * 32 * 3, 1)
        # self.lq = nn.Linear(
        #     int(hw / 8) * int(hw / 8) * 8 * dim + 32 * 32 * 3, 1)
        # self.Square = Square()

    def forward(self, x_in):
        x_in = x_in.view(-1, self.inchan, self.hw, self.hw)

        output1 = self.conv1(x_in)
        output2 = self.rb1(output1)
        output3 = self.rbc1(output2)
        output4 = self.rbc11(output3)
        output5 = self.rb2(output4)
        output6 = self.rbc2(output5)
        output7 = self.rbc22(output6)
        output8 = self.rb3(output7)
        output9 = self.rbc3(output8)
        output10 = self.rbc33(output9)

        outputfc1 = self.final_fc1(output1)
        outputfc2 = self.final_fc2(output4)
        outputfc3 = self.final_fc3(output7)
        outputfc4 = self.final_fc4(output10)

        energy = outputfc1 + outputfc2 + outputfc3 + outputfc4
        return energy


class Res18_Quadratic_64x64(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res18_Quadratic_64x64, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )

        self.rb4 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc4 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 16),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc44 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 16),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 16) * int(hw / 16) * 8 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 16) * int(hw / 16) * 8 * dim, 1)
        self.lq = nn.Linear(int(hw / 16) * int(hw / 16) * 8 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.rb4(output)
        output = self.rbc4(output)
        output = self.rbc44(output)
        output = output.view(-1, int(self.hw / 16) * int(self.hw / 16) * 8 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        # print(output.shape)
        # exit()
        return output


class Res18_Quadratic_MNIST(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res18_Quadratic_MNIST, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.lq = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in.view(-1, 1, 28, 28)
        output = F.pad(output, [2, 2, 2, 2])
        output = torch.cat([output, output, output], 1)
        # print(output.shape)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = output.view(-1, int(self.hw / 8) * int(self.hw / 8) * 8 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class Res12_Quadratic_MNIST(nn.Module):
    # 6 block resnet used in MIT EBM papaer for conditional Imagenet 32X32
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res12_Quadratic_MNIST, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.lq = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in
        output = F.pad(output, [2, 2, 2, 2])
        output = torch.cat([output, output, output], 1)
        # print(output.shape)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = output.view(-1, int(self.hw / 8) * int(self.hw / 8) * 8 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class MLP_Quadratic_MNIST(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(MLP_Quadratic_MNIST, self).__init__()

        self.AF = AF
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)

        self.ln1 = nn.Linear(200, 1)
        self.ln2 = nn.Linear(200, 1)
        self.lq = nn.Linear(200, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in.view(-1, 784)
        output = self.AF(self.fc1(output))
        output = self.AF(self.fc2(output))
        output = self.AF(self.fc3(output))

        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class Res12_Quadratic(nn.Module):
    # 6 block resnet used in MIT EBM papaer for conditional Imagenet 32X32
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res12_Quadratic, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.lq = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = output.view(-1, int(self.hw / 8) * int(self.hw / 8) * 8 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class Res6_Quadratic(nn.Module):
    # 3 block resnet for small MNIST experiment
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res6_Quadratic, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.ln1 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.lq = nn.Linear(int(hw / 8) * int(hw / 8) * 8 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        x_in = x_in.view(-1, self.inchan, self.hw, self.hw)
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = output.view(-1, int(self.hw / 8) * int(self.hw / 8) * 8 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class Res34_Quadratic(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False, AF=None):
        super(Res34_Quadratic, self).__init__()
        # made first layer 2*dim wide to not provide bottleneck

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 3)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc222 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc333 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3333 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33333 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb4 = ResidualBlock(
            8 * dim,
            16 * dim,
            3,
            int(hw / 8),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc4 = ResidualBlock(
            16 * dim,
            16 * dim,
            3,
            int(hw / 16),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc44 = ResidualBlock(
            16 * dim,
            16 * dim,
            3,
            int(hw / 16),
            resample="none",
            normalize=normalize,
            AF=AF,
        )

        self.ln1 = nn.Linear(int(hw / 16) * int(hw / 16) * 16 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 16) * int(hw / 16) * 16 * dim, 1)
        self.lq = nn.Linear(int(hw / 16) * int(hw / 16) * 16 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rbc222(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.rbc333(output)
        output = self.rbc3333(output)
        output = self.rbc33333(output)
        output = self.rb4(output)
        output = self.rbc4(output)
        output = self.rbc44(output)
        output = output.view(-1, int(self.hw / 16) * int(self.hw / 16) * 16 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        output = output.view(-1)
        return output


class Res34_Quadratic_Imagenet(nn.Module):
    # Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw=224, normalize=False, AF=None):
        super(Res34_Quadratic_Imagenet, self).__init__()
        # made first layer 2*dim wide to not provide bottleneck

        self.hw = hw
        hw = hw / 2
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan, dim, 5, 2)
        self.rb1 = ResidualBlock(
            dim, 2 * dim, 3, int(hw), resample="down", normalize=normalize, AF=AF
        )
        self.rbc1 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc11 = ResidualBlock(
            2 * dim,
            2 * dim,
            3,
            int(hw / 2),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb2 = ResidualBlock(
            2 * dim,
            4 * dim,
            3,
            int(hw / 2),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc2 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc22 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc222 = ResidualBlock(
            4 * dim,
            4 * dim,
            3,
            int(hw / 4),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb3 = ResidualBlock(
            4 * dim,
            8 * dim,
            3,
            int(hw / 4),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc333 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc3333 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc33333 = ResidualBlock(
            8 * dim,
            8 * dim,
            3,
            int(hw / 8),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rb4 = ResidualBlock(
            8 * dim,
            16 * dim,
            3,
            int(hw / 8),
            resample="down",
            normalize=normalize,
            AF=AF,
        )
        self.rbc4 = ResidualBlock(
            16 * dim,
            16 * dim,
            3,
            int(hw / 16),
            resample="none",
            normalize=normalize,
            AF=AF,
        )
        self.rbc44 = ResidualBlock(
            16 * dim,
            16 * dim,
            3,
            int(hw / 16),
            resample="none",
            normalize=normalize,
            AF=AF,
        )

        self.ln1 = nn.Linear(int(hw / 16) * int(hw / 16) * 16 * dim, 1)
        self.ln2 = nn.Linear(int(hw / 16) * int(hw / 16) * 16 * dim, 1)
        self.lq = nn.Linear(int(hw / 16) * int(hw / 16) * 16 * dim, 1)
        self.Square = Square()

    def forward(self, x_in):
        output = x_in.view(-1, self.inchan, self.hw, self.hw)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rbc222(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.rbc333(output)
        output = self.rbc3333(output)
        output = self.rbc33333(output)
        output = self.rb4(output)
        output = self.rbc4(output)
        output = self.rbc44(output)
        # print(output.shape)
        output = output.view(-1, int(self.hw / 32) * int(self.hw / 32) * 16 * self.dim)
        output = self.ln1(output) * self.ln2(output) + self.lq(self.Square(output))
        # print(output.shape)
        output = output.view(-1)
        # print(output.shape)
        return output
