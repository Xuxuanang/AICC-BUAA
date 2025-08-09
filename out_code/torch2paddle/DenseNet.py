import paddle
import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
import torch.nn
import torch.nn.functional
import torch.utils.checkpoint
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
__all__ = ['DenseNet', 'DenseNet121_Weights', 'DenseNet161_Weights', 'DenseNet169_Weights', 'DenseNet201_Weights', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

class _DenseLayer(paddle.nn.Layer):

    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool=False) -> None:
        super().__init__()
        self.norm1 = paddle.nn.BatchNorm2D()
        self.relu1 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(num_input_features, False, stride=bn_size * growth_rate, kernel_size=1)
        self.norm2 = paddle.nn.BatchNorm2D()
        self.relu2 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(bn_size * growth_rate, False, padding=1, stride=growth_rate, kernel_size=3)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = paddle.concat(x=inputs)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused    # 当前算子不支持转换
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:

        def closure(*inputs):
            return self.bn_function(inputs)
        return paddle.distributed.fleet.utils.recompute()

    @torch.jit._overload_method    # 当前算子不支持转换
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method    # 当前算子不支持转换
    def forward(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():    # 当前算子不支持转换
                raise Exception('Memory Efficient not supported in JIT')
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = paddle.nn.functional.dropout(self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(paddle.nn.LayerDict):
    _version = 2

    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float, memory_efficient: bool=False) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for (name, layer) in self.items():
            new_features = layer(features)
            features.append(new_features)
        return paddle.concat(x=features)

class _Transition(paddle.nn.Sequential):

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = paddle.nn.BatchNorm2D()
        self.relu = paddle.nn.ReLU()
        self.conv = paddle.nn.Conv2D(num_input_features, False, stride=num_output_features, kernel_size=1)
        self.pool = paddle.nn.AvgPool2D(2, stride=2)

class DenseNet(paddle.nn.Layer):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, growth_rate: int=32, block_config: Tuple[int, int, int, int]=(6, 12, 24, 16), num_init_features: int=64, bn_size: int=4, drop_rate: float=0, num_classes: int=1000, memory_efficient: bool=False) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = paddle.nn.Sequential(OrderedDict([('conv0', torch.nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', torch.nn.BatchNorm2d(num_init_features)), ('relu0', torch.nn.ReLU(inplace=True)), ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))    # 当前算子不支持转换
        num_features = num_init_features
        for (i, num_layers) in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', torch.nn.BatchNorm2d(num_features))    # 当前算子不支持转换
        self.classifier = paddle.nn.Linear()
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):    # 当前算子不支持转换
                paddle.nn.initializer.KaimingNormal()
            elif isinstance(m, torch.nn.BatchNorm2d):    # 当前算子不支持转换
                paddle.nn.initializer.Constant()
                paddle.nn.initializer.Constant()
            elif isinstance(m, torch.nn.Linear):    # 当前算子不支持转换
                paddle.nn.initializer.Constant()

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = paddle.nn.functional.relu()
        out = paddle.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = paddle.flatten()
        out = self.classifier(out)
        return out

def _load_state_dict(model: torch.nn.Module, weights: WeightsEnum, progress: bool) -> None:    # 当前算子不支持转换
    pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
    state_dict = weights.get_state_dict(progress=progress, check_hash=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

def _densenet(growth_rate: int, block_config: Tuple[int, int, int, int], num_init_features: int, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)
    return model
_COMMON_META = {'min_size': (29, 29), 'categories': _IMAGENET_CATEGORIES, 'recipe': 'https://github.com/pytorch/vision/pull/116', '_docs': 'These weights are ported from LuaTorch.'}

class DenseNet121_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet121-a639ec97.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 7978856, '_metrics': {'ImageNet-1K': {'acc@1': 74.434, 'acc@5': 91.972}}, '_ops': 2.834, '_file_size': 30.845})
    DEFAULT = IMAGENET1K_V1

class DenseNet161_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet161-8d451a50.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 28681000, '_metrics': {'ImageNet-1K': {'acc@1': 77.138, 'acc@5': 93.56}}, '_ops': 7.728, '_file_size': 110.369})
    DEFAULT = IMAGENET1K_V1

class DenseNet169_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet169-b2777c0a.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 14149480, '_metrics': {'ImageNet-1K': {'acc@1': 75.6, 'acc@5': 92.806}}, '_ops': 3.36, '_file_size': 54.708})
    DEFAULT = IMAGENET1K_V1

class DenseNet201_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet201-c1103571.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 20013928, '_metrics': {'ImageNet-1K': {'acc@1': 76.896, 'acc@5': 93.37}}, '_ops': 4.291, '_file_size': 77.373})
    DEFAULT = IMAGENET1K_V1

@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet121_Weights.IMAGENET1K_V1))
def densenet121(*, weights: Optional[DenseNet121_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    """Densenet-121 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet121_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet121_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet121_Weights
        :members:
    """
    weights = DenseNet121_Weights.verify(weights)
    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)

@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet161_Weights.IMAGENET1K_V1))
def densenet161(*, weights: Optional[DenseNet161_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    """Densenet-161 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet161_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet161_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet161_Weights
        :members:
    """
    weights = DenseNet161_Weights.verify(weights)
    return _densenet(48, (6, 12, 36, 24), 96, weights, progress, **kwargs)

@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet169_Weights.IMAGENET1K_V1))
def densenet169(*, weights: Optional[DenseNet169_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    """Densenet-169 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet169_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet169_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet169_Weights
        :members:
    """
    weights = DenseNet169_Weights.verify(weights)
    return _densenet(32, (6, 12, 32, 32), 64, weights, progress, **kwargs)

@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet201_Weights.IMAGENET1K_V1))
def densenet201(*, weights: Optional[DenseNet201_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    """Densenet-201 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet201_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet201_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet201_Weights
        :members:
    """
    weights = DenseNet201_Weights.verify(weights)
    return _densenet(32, (6, 12, 48, 32), 64, weights, progress, **kwargs)