import numpy
import mindspore
from models import img_network
from models import common_network

def get_fea(args):
    net = img_network.ResBase('resnet50')
    return net

class Resnet50(mindspore.nn.Cell):
    def __init__(self, args):
        super(Resnet50, self).__init__()
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(args.num_classes, self.featurizer.in_features)

        self.network = mindspore.nn.SequentialCell(self.featurizer, self.classifier)

    def construct(self, x):
        return self.network(x)