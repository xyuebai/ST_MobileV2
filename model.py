from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SpatialTransformer(nn.Module):
    def __init__(self, shearing=True):
        super(SpatialTransformer, self).__init__()
        self.shearing = shearing

        self.localization = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(2, stride=2),
            # nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
        )
        self.localization[0].weight.data.zero_()
        if self.shearing:
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                # nn.Linear(3 * 112 * 112, 6),
                # nn.ReLU(True),
                nn.Linear(3 * 112 * 112, 3 * 2)
            )
            # Initialize the weights/bias with identity transformation

            self.fc_loc[0].weight.data.zero_()
            self.fc_loc[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc = nn.Sequential(
                # nn.Linear(3 * 112 * 112, 6),
                # nn.ReLU(True),
                nn.Linear(3 * 112 * 112, 2 * 2)
            )
            # Initialize the weights/bias with identity transformation

            self.fc_loc[0].weight.data.zero_()
            self.fc_loc[0].bias.data.copy_(torch.tensor([1, 0, 0, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 3 * 112 * 112)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def oshear_stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 3 * 112 * 112)
        transformer = self.fc_loc(xs)

        transformer = transformer.view(-1, 4)
        theta = torch.zeros(transformer.size()[0], 2, 3)
        for i in range(transformer.size()[0]):
            scale = (transformer[i][0] ** 2 + transformer[i][1] ** 2) ** (1 / 2)
            theta[i][0][0] = transformer[i][0]
            theta[i][0][1] = -transformer[i][1]
            theta[i][0][2] = transformer[i][2]
            theta[i][1][0] = transformer[i][1]
            theta[i][1][1] = transformer[i][0]
            theta[i][1][2] = transformer[i][2]
            np_theta = theta[i].detach().numpy()
            s = np.linalg.eigvals(np_theta[:, :-1])
            R = s.max() / s.min()

            if R >= 2.0:
                theta[i][0][0] = 1.0
                theta[i][0][1] = 0.0
                theta[i][0][2] = 0.0
                theta[i][1][0] = 0.0
                theta[i][1][1] = 1.0
                theta[i][1][2] = 0.0

        theta = theta.cuda()
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        if self.shearing:
            x = self.stn(x)
        else:
            x = self.oshear_stn(x)
        return x


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class STN_MobileNet2(nn.Module):
    """STN_MobileNet2 implementation.
    """

    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=101, activation=nn.ReLU6,
                 shearing=True):
        """
        STN_MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(STN_MobileNet2, self).__init__()

        self.scale = scale
        self.shearing = shearing
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes
        self.input_size = input_size

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        # assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()
        self.init_params()

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)  # confirmed by paper authors
        self.fc = nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.stnmod = SpatialTransformer(self.shearing)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.stnmod(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # TODO not needed(?)


if __name__ == "__main__":
    """Testing
    """

    model = STN_MobileNet2().cuda()

    print(model)

    # test network with a single image
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    img_pil = Image.open("000045.jpg")
    imgplot = plt.imshow(img_pil)
    plt.show()
    img_tensor = transforms.ToTensor()(img_pil)
    img_tensor = img_tensor.view(1, 3, 224, 224)
    target = [25]  # a dummy target, for example
    target = torch.tensor(target)
    #    target = target.view(1, -1)  # make it the same shape as output
    img_tensor, target = img_tensor.cuda(), target.cuda()
    output = model(img_tensor)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()  # Does the update
    print(output)
    print(target)

    # loss = F.nll_loss(output, target)
    print(loss.item())

    img_stn_tensor = model.stnmod(img_tensor).cpu()
    img_stn_tensor = img_stn_tensor.view(3, 224, 224)
    img_PIL = transforms.ToPILImage()(img_stn_tensor)
    imgplot = plt.imshow(img_PIL)
    plt.show()

    # model2 = MobileNet2(scale=0.35)
    # print(model2)
    # model3 = MobileNet2(in_channels=2, num_classes=10)
    # print(model3)
    # x = torch.randn(1, 2, 224, 224)
    # print(model3(x))
    # model4_size = 32 * 10
    # model4 = MobileNet2(input_size=model4_size, num_classes=10)
    # print(model4)
    # x2 = torch.randn(1, 3, model4_size, model4_size)
    # print(model4(x2))
    # model5 = MobileNet2(input_size=196, num_classes=10)
    # x3 = torch.randn(1, 3, 196, 196)
    # print(model5(x3))  # fail
