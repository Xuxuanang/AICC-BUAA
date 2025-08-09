import torch

class Bottleneck(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 1, stride[0], padding[0], bias=False, padding_mode='zeros'), torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(), torch.nn.Conv2d(out_channels, out_channels, 3, stride[1], padding[1], bias=False, padding_mode='zeros'), torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(), torch.nn.Conv2d(out_channels, out_channels * 4, 1, stride[2], padding[2], bias=False, padding_mode='zeros'), torch.nn.BatchNorm2d(out_channels * 4))
        self.shortcut = torch.nn.Sequential()
        if first:
            self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels * 4, 1, stride[1], bias=False, padding_mode='zeros'), torch.nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class ResNet50(torch.nn.Module):

    def __init__(self, Bottleneck, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 1, 1, 0, bias=False, padding_mode='zeros'), torch.nn.BatchNorm2d(64), torch.nn.MaxPool2d(3, 2))
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)
        self.avgpool = torch.nn.MaxPool2d((7, 7))
        self.fc = torch.nn.Linear(2048, num_classes)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.mean(-1).mean(-1)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class Resnet50_model(torch.nn.Module):

    def __init__(self, args):
        super(Resnet50_model, self).__init__()
        self.model = ResNet50(Bottleneck, num_classes=args.num_classes)

    def forward(self, x):
        return self.model(x)