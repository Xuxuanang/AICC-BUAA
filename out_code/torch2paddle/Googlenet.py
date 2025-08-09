import paddle
import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn
import torch.nn.functional
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
__all__ = ['GoogLeNet', 'GoogLeNetOutputs', '_GoogLeNetOutputs', 'GoogLeNet_Weights', 'googlenet']
GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor], 'aux_logits1': Optional[Tensor]}
_GoogLeNetOutputs = GoogLeNetOutputs

class GoogLeNet(paddle.nn.Layer):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes: int=1000, aux_logits: bool=True, transform_input: bool=False, init_weights: Optional[bool]=None, blocks: Optional[List[Callable[..., torch.nn.Module]]]=None, dropout: float=0.2, dropout_aux: float=0.7) -> None:    # 当前算子不支持转换
        super().__init__()
        _log_api_usage_once(self)
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(f'blocks length should be 3 instead of {len(blocks)}')
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = paddle.nn.MaxPool2D(2, 3)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = paddle.nn.MaxPool2D(2, 3)
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = paddle.nn.MaxPool2D(2, 3)
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = paddle.nn.MaxPool2D(2, 2)
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None
        self.avgpool = paddle.nn.AdaptiveAvgPool2D((1, 1))
        self.dropout = paddle.nn.Dropout(dropout)
        self.fc = paddle.nn.Linear()
        if init_weights:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):    # 当前算子不支持转换
                    paddle.nn.initializer.TruncatedNormal(m.weight, 0.01, 2, 0.0)
                elif isinstance(m, torch.nn.BatchNorm2d):    # 当前算子不支持转换
                    paddle.nn.initializer.Constant()
                    paddle.nn.initializer.Constant()

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = paddle.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = paddle.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = paddle.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = paddle.concat(x=('1x_ch0', '1x_ch1', '1x_ch2'))
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = paddle.flatten()
        x = self.dropout(x)
        x = self.fc(x)
        return (x, aux2, aux1)

    @torch.jit.unused    # 当前算子不支持转换
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        (x, aux2, aux1) = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():    # 当前算子不支持转换
            if not aux_defined:
                warnings.warn('Scripted GoogleNet always returns GoogleNetOutputs Tuple')
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

class Inception(paddle.nn.Layer):

    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, ch5x5red: int, ch5x5: int, pool_proj: int, conv_block: Optional[Callable[..., torch.nn.Module]]=None) -> None:    # 当前算子不支持转换
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = paddle.nn.Sequential(conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))
        self.branch3 = paddle.nn.Sequential(conv_block(in_channels, ch5x5red, kernel_size=1), conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1))
        self.branch4 = paddle.nn.Sequential(paddle.nn.MaxPool2D(1, True, kernel_size=3, padding=1), conv_block(in_channels, pool_proj, kernel_size=1))

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs)

class InceptionAux(paddle.nn.Layer):

    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., torch.nn.Module]]=None, dropout: float=0.7) -> None:    # 当前算子不支持转换
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = paddle.nn.Linear()
        self.fc2 = paddle.nn.Linear()
        self.dropout = paddle.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = paddle.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = paddle.flatten()
        x = paddle.nn.functional.relu()
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BasicConv2d(paddle.nn.Layer):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channels, False, stride=out_channels)
        self.bn = paddle.nn.BatchNorm2D(0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return paddle.nn.functional.relu()

class GoogLeNet_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/googlenet-1378be20.pth', transforms=partial(ImageClassification, crop_size=224), meta={'num_params': 6624904, 'min_size': (15, 15), 'categories': _IMAGENET_CATEGORIES, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#googlenet', '_metrics': {'ImageNet-1K': {'acc@1': 69.778, 'acc@5': 89.53}}, '_ops': 1.498, '_file_size': 49.731, '_docs': 'These weights are ported from the original paper.'})
    DEFAULT = IMAGENET1K_V1

@register_model()
@handle_legacy_interface(weights=('pretrained', GoogLeNet_Weights.IMAGENET1K_V1))
def googlenet(*, weights: Optional[GoogLeNet_Weights]=None, progress: bool=True, **kwargs: Any) -> GoogLeNet:
    """GoogLeNet (Inception v1) model architecture from
    `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

    Args:
        weights (:class:`~torchvision.models.GoogLeNet_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.GoogLeNet_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.GoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
    """
    weights = GoogLeNet_Weights.verify(weights)
    original_aux_logits = kwargs.get('aux_logits', False)
    if weights is not None:
        if 'transform_input' not in kwargs:
            _ovewrite_named_param(kwargs, 'transform_input', True)
        _ovewrite_named_param(kwargs, 'aux_logits', True)
        _ovewrite_named_param(kwargs, 'init_weights', False)
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
    model = GoogLeNet(**kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        else:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them')
    return model