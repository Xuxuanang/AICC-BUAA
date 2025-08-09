import mindspore
class Bottleneck(mindspore.nn.Cell):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()