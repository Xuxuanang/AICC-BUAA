import paddle
import torch
import torch.nn as nn

class VGG11(paddle.nn.Layer):

    def __init__(self, in_channels, num_classes=10):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = paddle.nn.Sequential(paddle.nn.Conv2D(kernel_size=3, out_channels=self.in_channels, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, stride=2), paddle.nn.Conv2D(kernel_size=3, out_channels=64, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, stride=2), paddle.nn.Conv2D(kernel_size=3, out_channels=128, padding=1), paddle.nn.ReLU(), paddle.nn.Conv2D(kernel_size=3, out_channels=256, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, stride=2), paddle.nn.Conv2D(kernel_size=3, out_channels=256, padding=1), paddle.nn.ReLU(), paddle.nn.Conv2D(kernel_size=3, out_channels=512, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, stride=2), paddle.nn.Conv2D(kernel_size=3, out_channels=512, padding=1), paddle.nn.ReLU(), paddle.nn.Conv2D(kernel_size=3, out_channels=512, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, stride=2))
        self.linear_layers = paddle.nn.Sequential(paddle.nn.Linear(512 * 7 * 7, 4096), paddle.nn.ReLU(), paddle.nn.Dropout(0.5), paddle.nn.Linear(4096, 4096), paddle.nn.ReLU(), paddle.nn.Dropout(0.5), paddle.nn.Linear(4096, self.num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
if __name__ == '__main__':
    seed = 2024
    paddle.seed()
    vgg11 = VGG11(in_channels=3)
    output = vgg11(x)
    print(output.shape)
    print(output)