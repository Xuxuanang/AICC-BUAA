import API
import math
import mindspore


class Downblock(mindspore.nn.Cell):
    def __init__(self, channels, kernel_size=3):
        super(Downblock, self).__init__()
        self.dwconv = mindspore.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=2, pad_mode="pad", padding=1, dilation=1, group=channels, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")

        self.bn = mindspore.nn.BatchNorm2d(num_features=channels, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW")

    def construct(self, x):
        return self.bn(self.dwconv(x))


class GEBlock(mindspore.nn.Cell):
    def __init__(self, in_planes, out_planes, stride, spatial, extent=0, extra_params=True, mlp=True, dropRate=0.0):
        super(GEBlock, self).__init__()

        self.bnrelu = mindspore.nn.SequentialCell(mindspore.nn.BatchNorm2d(num_features=in_planes, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU())

        self.conv = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, pad_mode="pad", padding=1, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.BatchNorm2d(num_features=out_planes, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones", beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None, data_format="NCHW"), mindspore.nn.ReLU(), mindspore.nn.Dropout(p=dropRate), mindspore.nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, pad_mode="pad", padding=1, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"))

        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = (not self.equalInOut) and mindspore.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW") or None

        if extra_params:
            if extent: modules = [Downblock(out_planes)]
            for i in range((extent - 1) // 2):
                modules.append(mindspore.nn.SequentialCell(mindspore.nn.ReLU(), Downblock(out_planes)))
            self.downop = mindspore.nn.SequentialCell(*modules) if extent else Downblock(out_planes, kernel_size=spatial)
        else:
            self.downop = mindspore.ops.AdaptiveAvgPool2D(output_size=spatial // extent) if extent else mindspore.ops.AdaptiveAvgPool2D(output_size=1)

        self.mlp = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels=out_planes, out_channels=out_planes // 16, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"), mindspore.nn.ReLU(), mindspore.nn.Conv2d(in_channels=out_planes // 16, out_channels=out_planes, kernel_size=1, stride=1, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")) if mlp else lambda \
                x: x

    def construct(self, x):
        bnrelu = self.bnrelu(x)
        out = self.conv(bnrelu)
        map = self.mlp(self.downop(out))
        map = mindspore.ops.interpolate(map, out.shape[-1])
        if not self.equalInOut: x = self.convShortcut(bnrelu)
        return mindspore.ops.add(x, out * mindspore.ops.sigmoid(map))


class NetworkBlock(mindspore.nn.Cell):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, spatial, extent, extra_params, mlp,dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, spatial, extent, extra_params,mlp, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, spatial, extent, extra_params, mlp,dropRate):
        layers = []
        for i in range(int(nb_layers)):
            if i == 0:
                layers.append(block(in_planes, out_planes, stride, spatial, extent,extra_params, mlp,dropRate))
            else:
                layers.append(block(out_planes, out_planes, 1, spatial, extent,extra_params, mlp,dropRate))
        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        return self.layer(x)


class GENet(mindspore.nn.Cell):
    def __init__(self, num_classes=1000, extent=0, extra_params=True, mlp=True, dropRate=0.0):
        super(GENet, self).__init__()

        layer_nums = [3, 4, 6, 3]
        in_channels = [64, 256, 512, 1024]
        out_channels = [256, 512, 1024, 2048]
        self.out_channels = out_channels
        strides = [1, 2, 2, 2]
        spatial = [56, 28, 14, 7]

        block = GEBlock

        self.conv1 = mindspore.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, pad_mode="pad", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")
        self.max_pool1 = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid", data_format="NCHW")


        self.block1 = NetworkBlock(layer_nums[0], in_channels[0], out_channels[0], block, 1, spatial=spatial[0],extent=extent,extra_params=extra_params, mlp=mlp, dropRate=dropRate)
        self.block2 = NetworkBlock(layer_nums[1], in_channels[1], out_channels[1], block, 2, spatial=spatial[1],extent=extent,extra_params=extra_params, mlp=mlp, dropRate=dropRate)
        self.block3 = NetworkBlock(layer_nums[2], in_channels[2], out_channels[2], block, 2, spatial=spatial[2],extent=extent,extra_params=extra_params, mlp=mlp, dropRate=dropRate)

        self.block4 = NetworkBlock(layer_nums[3], in_channels[3], out_channels[3], block, 2, spatial=spatial[3],extent=extent,extra_params=extra_params, mlp=mlp, dropRate=dropRate)

        self.avg_pool = mindspore.ops.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = mindspore.nn.Dense(in_channels=out_channels[3], out_channels=num_classes, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)

        for m in self.cells():
            if isinstance(m, mindspore.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Normal(0, math.sqrt(2. / n)), m.weight.shape, m.weight.dtype))
            elif isinstance(m, mindspore.nn.BatchNorm2d):
                m.weight.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Constant(1), m.weight.shape, m.weight.dtype))
                m.bias.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Zero(), m.bias.shape, m.bias.dtype))
            elif isinstance(m, mindspore.nn.Dense):
                m.bias.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x):
        out = self.conv1(x)
        out = self.max_pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.avg_pool(out)
        out = out.view(-1, self.out_channels[3])
        return self.fc(out)


class GENet_Res50_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(GENet_Res50_model, self).__init__()
        self.model = GENet(args.num_classes, extra_params=True, mlp=False)

    def construct(self, x):
        return self.model(x)
