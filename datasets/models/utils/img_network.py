import mindspore

vgg_dict = {"vgg11": mindsporevision.models.vgg11, "vgg13": torchvision.models.vgg13, "vgg16": torchvision.models.vgg16, "vgg19": torchvision.models.vgg19,
            "vgg11bn": mindsporevision.models.vgg11_bn, "vgg13bn": torchvision.models.vgg13_bn, "vgg16bn": torchvision.models.vgg16_bn, "vgg19bn": torchvision.models.vgg19_bn}


class VGGBase(mindspore.nn.Cell):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = mindspore.nn.SequentialCell()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def construct(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": mindsporevision.models.resnet18, "resnet34": torchvision.models.resnet34, "resnet50": torchvision.models.resnet50,
            "resnet101": mindsporevision.models.resnet101, "resnet152": torchvision.models.resnet152, "resnext50": torchvision.models.resnext50_32x4d, "resnext101": torchvision.models.resnext101_32x8d}


class ResBase(mindspore.nn.Cell):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class DTNBase(mindspore.nn.Cell):
    def __init__(self):
        super(DTNBase, self).__init__()
# torch.nn.Dropout2d
        self.in_features = 256*4*4

    def construct(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(mindspore.nn.Cell):
    def __init__(self):
        super(LeNetBase, self).__init__()
# torch.nn.Dropout2d
        self.in_features = 50*4*4

    def construct(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x
