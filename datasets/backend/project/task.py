import torch

m = torch.nn.MaxPool2d(3, stride=2, padding=1)
input_x = torch.randn(20, 16, 50, 32)
output = m(input_x)
print(output.shape)