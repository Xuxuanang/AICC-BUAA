import mindspore
import torch
import torch.nn as nn

class VGG11(mindspore.nn.Cell):

    def __init__(self, in_channels, num_classes=10):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.linear_layers = nn.Sequential(nn.Linear(in_features=512 * 7 * 7, out_features=4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(in_features=4096, out_features=self.num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
if __name__ == '__main__':
    seed = 2024
    torch.manual_seed(seed)# 当前算子不支持转换
    vgg11 = VGG11(in_channels=3)
    output = vgg11(x)
    print(output.shape)
    print(output)