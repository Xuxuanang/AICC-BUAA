import API
import numpy
import mindspore
import sys


class Bottleneck(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                          kernel_size=1, stride=stride[0], pad_mode="pad",
                                                                          padding=padding[0], dilation=1, group=1, has_bias=False,
                                                                          weight_init="normal", bias_init="zeros", data_format="NCHW"),
                                                      mindspore.nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9,
                                                                               affine=True, gamma_init="ones", beta_init="zeros",
                                                                               moving_mean_init="zeros", moving_var_init="ones",
                                                                               use_batch_statistics=None, data_format="NCHW"),
                                                      mindspore.nn.ReLU(), mindspore.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                                                               kernel_size=3, stride=stride[1], pad_mode="pad",
                                                                                               padding=padding[1], dilation=1, group=1,
                                                                                               has_bias=False, weight_init="normal",
                                                                                               bias_init="zeros", data_format="NCHW"),
                                                      mindspore.nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9,
                                                                               affine=True, gamma_init="ones", beta_init="zeros",
                                                                               moving_mean_init="zeros", moving_var_init="ones",
                                                                               use_batch_statistics=None, data_format="NCHW"),
                                                      mindspore.nn.ReLU(), mindspore.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4,
                                                                                               kernel_size=1, stride=stride[2], pad_mode="pad",
                                                                                               padding=padding[2], dilation=1, group=1,
                                                                                               has_bias=False, weight_init="normal",
                                                                                               bias_init="zeros", data_format="NCHW"),
                                                      mindspore.nn.BatchNorm2d(num_features=out_channels * 4, eps=1e-5, momentum=0.9,
                                                                               affine=True, gamma_init="ones", beta_init="zeros",
                                                                               moving_mean_init="zeros", moving_var_init="ones",
                                                                               use_batch_statistics=None, data_format="NCHW"))

        self.shortcut = mindspore.nn.SequentialCell()
        if first:
            self.shortcut = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride[1], pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=out_channels * 4, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"))

    def construct(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = mindspore.ops.ReLU()(out)
        return out


class ResNet50(mindspore.nn.Cell):
    def __init__(self, Bottleneck, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid", data_format="NCHW"))

        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = mindspore.nn.MaxPool2d(kernel_size=(7, 7), stride=1, pad_mode="valid", data_format="NCHW")
        self.fc = mindspore.nn.Dense(in_channels=2048, out_channels=num_classes, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.mean(-1).mean(-1)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class Resnet50_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(Resnet50_model, self).__init__()
        self.model = ResNet50(Bottleneck, num_classes=args.num_classes)

    def construct(self, x):
        return self.model(x)



