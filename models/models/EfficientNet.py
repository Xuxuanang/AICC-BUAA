import API
import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import mindspore


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob  
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + mindspore.ops.UniformReal()(0, 0, shape)
    random_tensor.floor()  
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(mindspore.nn.Cell):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(mindspore.nn.SequentialCell):
    def __init__(self,in_planes: int,out_planes: int,kernel_size: int = 3,stride: int = 1,groups: int = 1,norm_layer: Optional[Callable[..., mindspore.nn.Cell]] = None,activation_layer: Optional[Callable[..., mindspore.nn.Cell]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = mindspore.nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = mindspore.nn.SiLU

        super(ConvBNActivation, self).__init__(mindspore.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, pad_mode="pad", padding=padding, dilation=1, group=groups, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"),norm_layer(out_planes),activation_layer())


class SqueezeExcitation(mindspore.nn.Cell):
    def __init__(self,input_c: int,expand_c: int,squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = mindspore.nn.Conv2d(in_channels=expand_c, out_channels=squeeze_c, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")
        self.ac1 = mindspore.nn.SiLU()
        self.fc2 = mindspore.nn.Conv2d(in_channels=squeeze_c, out_channels=expand_c, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")
        self.ac2 = mindspore.nn.Sigmoid()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        scale = mindspore.ops.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    def __init__(self,kernel: int,input_c: int,out_c: int,expanded_ratio: int,stride: int,use_se: bool,drop_rate: float,index: str,width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(mindspore.nn.Cell):
    def __init__(self,cnf: InvertedResidualConfig,norm_layer: Callable[..., mindspore.nn.Cell]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = mindspore.nn.SiLU

        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,cnf.expanded_c,kernel_size=1,norm_layer=norm_layer,activation_layer=activation_layer)})

        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,cnf.expanded_c,kernel_size=cnf.kernel,stride=cnf.stride,groups=cnf.expanded_c,norm_layer=norm_layer,activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,cnf.expanded_c)})

        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,cnf.out_c,kernel_size=1,norm_layer=norm_layer,activation_layer=mindspore.nn.Identity)})

        self.block = mindspore.nn.SequentialCell(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = mindspore.nn.Identity()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(mindspore.nn.Cell):
    def __init__(self,width_coefficient: float,depth_coefficient: float,num_classes: int = 1000,dropout_rate: float = 0.2,drop_connect_rate: float = 0.2,block: Optional[Callable[..., mindspore.nn.Cell]] = None,norm_layer: Optional[Callable[..., mindspore.nn.Cell]] = None):
        super(EfficientNet, self).__init__()

        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))




        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(mindspore.nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,width_coefficient=width_coefficient)

        bneck_conf = partial(InvertedResidualConfig,width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    cnf[-3] = 1
                    cnf[1] = cnf[2]

                cnf[-1] = args[-2] * b / num_blocks
                index = 'stage' + str(stage + 1) + chr(i + 97)
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        layers = OrderedDict()

        layers.update({"stem_conv": ConvBNActivation(in_planes=3,out_planes=adjust_channels(32),kernel_size=3,stride=2,norm_layer=norm_layer)})

        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,out_planes=last_conv_output_c,kernel_size=1,norm_layer=norm_layer)})

        self.features = mindspore.nn.SequentialCell(layers)
        self.avgpool = mindspore.ops.AdaptiveAvgPool2D(output_size=1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(mindspore.nn.Dropout(keep_prob=dropout_rate, dtype="mindspore.float32"))
        classifier.append(mindspore.nn.Dense(in_channels=last_conv_output_c, out_channels=num_classes, weight_init="normal", bias_init="zeros", has_bias=True, activation=None))
        self.classifier = mindspore.nn.SequentialCell(*classifier)

        for m in self.cells():
            if isinstance(m, mindspore.nn.Conv2d):
                mindspore.common.initializer.HeNormal()(0, "fan_out", "leaky_relu")
                if m.bias is not None:
                    mindspore.common.initializer.Zero()()
            elif isinstance(m, mindspore.nn.BatchNorm2d):
                mindspore.common.initializer.One()()
                mindspore.common.initializer.Zero()()
            elif isinstance(m, mindspore.nn.Dense):
                mindspore.common.initializer.Normal()(0.01, 0)
                mindspore.common.initializer.Zero()()

    def _forward_impl(self, x: mindspore.Tensor) -> mindspore.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = mindspore.ops.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    return EfficientNet(width_coefficient=1.0,depth_coefficient=1.0,dropout_rate=0.2,num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    return EfficientNet(width_coefficient=1.0,depth_coefficient=1.1,dropout_rate=0.2,num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    return EfficientNet(width_coefficient=1.1,depth_coefficient=1.2,dropout_rate=0.3,num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    return EfficientNet(width_coefficient=1.2,depth_coefficient=1.4,dropout_rate=0.3,num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    return EfficientNet(width_coefficient=1.4,depth_coefficient=1.8,dropout_rate=0.4,num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    return EfficientNet(width_coefficient=1.6,depth_coefficient=2.2,dropout_rate=0.4,num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    return EfficientNet(width_coefficient=1.8,depth_coefficient=2.6,dropout_rate=0.5,num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    return EfficientNet(width_coefficient=2.0,depth_coefficient=3.1,dropout_rate=0.5,num_classes=num_classes)


class Efficientnet_b3_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(Efficientnet_b3_model, self).__init__()
        self.model = efficientnet_b3(args.num_classes)

    def construct(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = efficientnet_b0()
    print(model)


# torchsummary.summary
