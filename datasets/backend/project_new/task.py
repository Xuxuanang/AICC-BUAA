import mindspore
m = mindspore.nn.MaxPool2d(3, 2, 'same')
input_x = torch.randn(20, 16, 50, 32)
output = m(input_x)
print(output.shape)