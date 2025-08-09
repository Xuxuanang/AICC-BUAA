import mindspore
from models.utils import init_weights


class feat_bottleneck(mindspore.nn.Cell):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = mindspore.nn.BatchNorm1d(num_features=bottleneck_dim, eps=1e-5, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None)
        self.relu = mindspore.nn.ReLU()
        self.dropout = mindspore.nn.Dropout(keep_prob=0.5, dtype="mindspore.float32")
        self.bottleneck = mindspore.nn.Dense(in_channels=feature_dim, out_channels=bottleneck_dim, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)
        self.type = type

    def construct(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(mindspore.nn.Cell):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
# torch.nn.utils.weight_norm
        else:
            self.fc = mindspore.nn.Dense(in_channels=bottleneck_dim, out_channels=class_num, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)

    def construct(self, x):
        x = self.fc(x)
        return x


class feat_classifier_two(mindspore.nn.Cell):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = mindspore.nn.Dense(in_channels=input_dim, out_channels=bottleneck_dim, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)
        self.fc1 = mindspore.nn.Dense(in_channels=bottleneck_dim, out_channels=class_num, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)

    def construct(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x
