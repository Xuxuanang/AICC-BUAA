import torch
from collections import OrderedDict
from typing import Any, Tuple
import mindspore

class _DenseLayer(mindspore.nn.Cell):    # 当前算子不支持转换

    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool=False) -> None:
        super().__init__()
        self.norm1: mindspore.nn.BatchNorm2d    # 当前算子不支持转换
        self.insert_child_to_cell('norm1', mindspore.nn.BatchNorm2d(num_features=num_input_features, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'))    # 当前算子不支持转换
        self.relu1: mindspore.nn.ReLU    # 当前算子不支持转换
        self.insert_child_to_cell('relu1', mindspore.nn.ReLU())    # 当前算子不支持转换
        self.conv1: mindspore.nn.Conv2d    # 当前算子不支持转换
        self.insert_child_to_cell('conv1', mindspore.nn.Conv2d(in_channels=num_input_features, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, pad_mode='pad', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCHW'))    # 当前算子不支持转换
        self.norm2: mindspore.nn.BatchNorm2d    # 当前算子不支持转换
        self.insert_child_to_cell('norm2', mindspore.nn.BatchNorm2d(num_features=bn_size * growth_rate, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'))    # 当前算子不支持转换
        self.relu2: mindspore.nn.ReLU    # 当前算子不支持转换
        self.insert_child_to_cell('relu2', mindspore.nn.ReLU())    # 当前算子不支持转换
        self.conv2: mindspore.nn.Conv2d    # 当前算子不支持转换
        self.insert_child_to_cell('conv2', mindspore.nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, pad_mode='pad', padding=1, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCHW'))    # 当前算子不支持转换
        self.drop_rate = float(drop_rate)

    def construct(self, input: mindspore.Tensor) -> mindspore.Tensor:    # 当前算子不支持转换
        prev_features = input
        new_features0 = mindspore.ops.Concat()(1)    # 当前算子不支持转换
        new_features1 = self.conv1(self.relu1(self.norm1(new_features0)))
        new_features2 = self.conv2(self.relu2(self.norm2(new_features1)))
        if self.drop_rate > 0:
            new_features2 = mindspore.nn.Dropout(keep_prob=self.drop_rate, dtype='mindspore.float32')(new_features2)    # 当前算子不支持转换
        return new_features2

class _DenseBlock(mindspore.nn.CellList):    # 当前算子不支持转换

    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.append(layer)

    def construct(self, init_features: mindspore.Tensor) -> mindspore.Tensor:    # 当前算子不支持转换
        features = [init_features]
        for layer in self:
            new_features = layer(features)
            features.append(new_features)
        return mindspore.ops.Concat()(1)    # 当前算子不支持转换

class _Transition(mindspore.nn.SequentialCell):    # 当前算子不支持转换

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.insert_child_to_cell('norm', mindspore.nn.BatchNorm2d(num_features=num_input_features, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'))    # 当前算子不支持转换
        self.insert_child_to_cell('relu', mindspore.nn.ReLU())    # 当前算子不支持转换
        self.insert_child_to_cell('conv', mindspore.nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=1, stride=1, pad_mode='pad', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCHW'))    # 当前算子不支持转换
        self.insert_child_to_cell('pool', mindspore.nn.AvgPool2d(kernel_size=1, stride=1, pad_mode='valid', data_format='NCHW'))    # 当前算子不支持转换

class DenseNet(mindspore.nn.Cell):    # 当前算子不支持转换

    def __init__(self, growth_rate: int=32, block_config: Tuple[int, int, int, int]=(6, 12, 24, 16), num_init_features: int=64, bn_size: int=4, drop_rate: float=0, num_classes: int=1000) -> None:
        super().__init__()
        self.features = mindspore.nn.SequentialCell(OrderedDict([('conv0', mindspore.nn.Conv2d(in_channels=3, out_channels=num_init_features, kernel_size=7, stride=2, pad_mode='pad', padding=3, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCHW')), ('norm0', mindspore.nn.BatchNorm2d(num_features=num_init_features, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')), ('relu0', mindspore.nn.ReLU()), ('pool0', mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid', data_format='NCHW'))]))    # 当前算子不支持转换
        num_features = num_init_features
        for (i, num_layers) in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.insert_child_to_cell('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.insert_child_to_cell('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.insert_child_to_cell('norm5', mindspore.nn.BatchNorm2d(num_features=num_features, eps=1e-05, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW'))    # 当前算子不支持转换
        self.classifier = mindspore.nn.Dense(in_channels=num_features, out_channels=num_classes, weight_init='normal', bias_init='zeros', has_bias=True, activation=None)    # 当前算子不支持转换
        for m in self.cells():
            if isinstance(m, mindspore.nn.Conv2d):    # 当前算子不支持转换
                mindspore.common.initializer('he_normal', m.weight)    # 当前算子不支持转换
            elif isinstance(m, mindspore.nn.BatchNorm2d):    # 当前算子不支持转换
                mindspore.common.initializer.Constant(1)(m.weight)    # 当前算子不支持转换
                mindspore.common.initializer.Constant(0)(m.bias)    # 当前算子不支持转换
            elif isinstance(m, mindspore.nn.Dense):    # 当前算子不支持转换
                mindspore.common.initializer.Constant(0)(m.bias)    # 当前算子不支持转换

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:    # 当前算子不支持转换
        features = self.features(x)
        out = torch.nn.functional.relu(out)
        out = mindspore.ops.adaptive_avg_pool2d(out, (1, 1))    # 当前算子不支持转换
        out = mindspore.ops.flatten(out, start_dim=1)    # 当前算子不支持转换
        out = self.classifier(out)
        return out

def densenet121(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 24, 16), 64, **kwargs)

def densenet161(**kwargs: Any) -> DenseNet:
    return DenseNet(48, (6, 12, 36, 24), 96, **kwargs)

def densenet169(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 32, 32), 64, **kwargs)

def densenet201(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 48, 32), 64, **kwargs)

class DenseNet121_model(mindspore.nn.Cell):    # 当前算子不支持转换

    def __init__(self, args):
        super(DenseNet121_model, self).__init__()
        self.model = densenet121(num_classes=args.num_classes)

    def construct(self, x):
        return self.model(x)
if __name__ == '__main__':
    from PIL import Image
    import re

    def resize_padding(image, target_length, value=0):
        (h, w) = image.size
        (ih, iw) = (target_length, target_length)
        scale = min(iw / w, ih / h)
        (nw, nh) = (int(scale * w), int(scale * h))
        image_resized = image.resize((nh, nw), Image.ANTIALIAS)
        image_paded = Image.new('RGB', (ih, iw), value)
        (dw, dh) = ((iw - nw) // 2, (ih - nh) // 2)
        image_paded.paste(image_resized, (dh, dw, nh + dh, nw + dw))
        return image_paded
    transform = mindspore.dataset.transforms.c_transforms.Compose(transforms=[mindspore.dataset.vision.py_transforms.ToTensor(output_type='np.float32'), mindspore.dataset.vision.c_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])    # 当前算子不支持转换
    image = resize_padding(Image.open('./car.jpg'), 224)
    image = transform(image)
    image = image.reshape(1, 3, 224, 224)
    weight_path = './checkpoint/densenet121-a639ec97.pth'
    pre_weights = mindspore.load_checkpoint(ckpt_file_name=weight_path, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode='AES-GCM')    # 当前算子不支持转换
    pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
    for key in list(pre_weights.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pre_weights[new_key] = pre_weights[key]
            del pre_weights[key]
    model = densenet121()
    model.load_state_dict(pre_weights)
    output = mindspore.ops.Squeeze()(())    # 当前算子不支持转换
    predict = mindspore.ops.softmax(output, axis=0)    # 当前算子不支持转换
    predict_cla = mindspore.ops.Argmax()(-1, 'mindspore.dtype.int32').numpy()    # 当前算子不支持转换
    print(predict_cla)
    print(predict[predict_cla])