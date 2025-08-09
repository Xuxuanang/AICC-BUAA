import mindspore

class Bottleneck(mindspore.nn.Cell):

    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = mindspore.nn.SequentialCell(
            mindspore.nn.Conv2d(in_channels, out_channels, 1, stride[0], pad_mode='pad', padding=padding[0], has_bias=False),
            mindspore.nn.BatchNorm2d(out_channels), mindspore.nn.ReLU(), mindspore.nn.Conv2d(out_channels, out_channels, 3, stride[1], pad_mode='pad', padding=padding[1], has_bias=False), mindspore.nn.BatchNorm2d(out_channels), mindspore.nn.ReLU(), mindspore.nn.Conv2d(out_channels, out_channels * 4, 1, stride[2], pad_mode='pad', padding=padding[2], has_bias=False), mindspore.nn.BatchNorm2d(out_channels * 4))
        self.shortcut = mindspore.nn.SequentialCell()
        if first:
            self.shortcut = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels, out_channels * 4, 1, stride[1], pad_mode='pad', has_bias=False), mindspore.nn.BatchNorm2d(out_channels * 4))

    def construct(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = mindspore.ops.ReLU()(out)
        return out

class ResNet50(mindspore.nn.Cell):

    def __init__(self, Bottleneck, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(3, 64, 1, 1, pad_mode='valid', padding=0, has_bias=False),
                                                 mindspore.nn.BatchNorm2d(64),
                                                 mindspore.nn.MaxPool2d(3, 2, 'same'))
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)
        self.avgpool = mindspore.nn.MaxPool2d((7, 7), pad_mode='valid')
        self.fc = mindspore.nn.Dense(2048, num_classes)

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