import API
from typing import List, Callable

import mindspore


def channel_shuffle(x: mindspore.Tensor, groups: int) -> mindspore.Tensor:
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = mindspore.ops.transpose(x, (0, 2, 1, 3, 4))

    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(mindspore.nn.Cell):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = mindspore.nn.SequentialCell(self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1), mindspore.nn.BatchNorm2d(num_features=input_c, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.Conv2d(in_channels=input_c, out_channels=branch_features, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=branch_features, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU())
        else:
            self.branch1 = mindspore.nn.SequentialCell()

        self.branch2 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=input_c if self.stride > 1 else branch_features, out_channels=branch_features, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=branch_features, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU(), self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1), mindspore.nn.BatchNorm2d(num_features=branch_features, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.Conv2d(in_channels=branch_features, out_channels=branch_features, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=branch_features, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU())

    @staticmethod
    def depthwise_conv(input_c: int,output_c: int,kernel_s: int,stride: int = 1,padding: int = 0,bias: bool = False) -> mindspore.nn.Conv2d:
        return mindspore.nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s, stride=stride, pad_mode="pad", padding=padding, dilation=1, group=input_c, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        if self.stride == 1:
            x1, x2 = mindspore.ops.Split(axis=1, output_num=2)(x)
            out = mindspore.ops.concat((x1, self.branch2(x2)), axis=1)
        else:
            out = mindspore.ops.concat((self.branch1(x), self.branch2(x)), axis=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(mindspore.nn.Cell):
    def __init__(self,stages_repeats: List[int],stages_out_channels: List[int],num_classes: int = 1000,inverted_residual: Callable[..., mindspore.nn.Cell] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2, pad_mode="pad", padding=1, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=output_channels, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU())
        input_channels = output_channels

        self.maxpool = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid", data_format="NCHW")

        self.stage2: mindspore.nn.Sequential
        self.stage3: mindspore.nn.Sequential
        self.stage4: mindspore.nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, mindspore.nn.SequentialCell(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=output_channels, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU())

        self.fc = mindspore.nn.Dense(in_channels=output_channels, out_channels=num_classes, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)

    def _forward_impl(self, x: mindspore.Tensor) -> mindspore.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],stages_out_channels=[24, 48, 96, 192, 1024],num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],stages_out_channels=[24, 116, 232, 464, 1024],num_classes=num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],stages_out_channels=[24, 176, 352, 704, 1024],num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],stages_out_channels=[24, 244, 488, 976, 2048],num_classes=num_classes)

    return model


class ShuffleNetV2_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(ShuffleNetV2_model, self).__init__()
        self.model = shufflenet_v2_x1_5(args.num_classes)

    def construct(self, x):
        return self.model(x)

