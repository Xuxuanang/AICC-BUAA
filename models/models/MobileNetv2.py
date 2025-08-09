import API
import mindspore


def _make_divisible(ch, divisor, min_ch=None):
    if not min_ch:
        min_ch = divisor

    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class Inverted_residual_bottleneck(mindspore.nn.Cell):
    def __init__(self, mul_ratio, inchannels, outchannels, stride):
        super(Inverted_residual_bottleneck, self).__init__()
        self.shortcut = True if stride == 1 and inchannels == outchannels else False

        layer = []
        if mul_ratio != 1:
            layer.extend([
                mindspore.nn.Conv2d(inchannels, inchannels * mul_ratio, kernel_size=1, stride=1),
                mindspore.nn.BatchNorm2d(inchannels * mul_ratio),
                mindspore.nn.ReLU6()
            ])

        layer.extend([
            mindspore.nn.Conv2d(inchannels * mul_ratio, inchannels * mul_ratio, kernel_size=3, stride=stride, pad_mode="pad", padding=1, group=inchannels * mul_ratio),
            mindspore.nn.BatchNorm2d(inchannels * mul_ratio),
            mindspore.nn.ReLU6(),
            mindspore.nn.Conv2d(inchannels * mul_ratio, outchannels, kernel_size=1, stride=1),
            mindspore.nn.BatchNorm2d(outchannels)
        ])
        self.conv = mindspore.nn.SequentialCell(*layer)

    def construct(self, x):
        if self.shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Mobilenet_V2(mindspore.nn.Cell):
    def __init__(self, inchannel, numclasses, alpha, round_nearest=8):
        super(Mobilenet_V2, self).__init__()
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        self.conv1 = mindspore.nn.Conv2d(in_channels=inchannel, out_channels=input_channel, kernel_size=3, stride=2, pad_mode="pad", padding=1, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")
        setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        self.block = Inverted_residual_bottleneck
        self.stages = mindspore.nn.CellList([])
        for t, c, n, s in setting:
            cout = _make_divisible(c * alpha, round_nearest)
            self.stages.append(self._make_stage(t, input_channel, cout, n, s))
            input_channel = cout
        self.conv2 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=input_channel, out_channels=last_channel, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=last_channel, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU())
        self.pool = mindspore.nn.MaxPool2d(kernel_size=7, stride=1, pad_mode="valid", data_format="NCHW")
        self.fc = mindspore.nn.SequentialCell(mindspore.nn.Dropout(keep_prob=0.2, dtype=mindspore.float32), mindspore.nn.Conv2d(in_channels=last_channel, out_channels=numclasses, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"))
        for m in self.cells():
            if isinstance(m, mindspore.nn.Conv2d):
                mindspore.common.initializer.Normal(0.01, 0)
                if m.bias is not None:
                    mindspore.common.initializer.Zero()
            elif isinstance(m, mindspore.nn.BatchNorm2d):
                mindspore.common.initializer.One()
                mindspore.common.initializer.Zero()

    def _make_stage(self, mult, inchannel, ouchannel, repeat, stride):
        strides = [stride] + [1] * repeat
        layers = []
        for i in range(repeat):
            layers.append(self.block(mult, inchannel, ouchannel, strides[i]))
            inchannel = ouchannel
        return mindspore.nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class MobileNetv2_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(MobileNetv2_model, self).__init__()
        self.model = Mobilenet_V2(3, args.num_classes, 0.5)

    def construct(self, x):
        return self.model(x)

