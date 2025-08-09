import mindspore
import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
# torch.nn.init.kaiming_uniform_
        mindspore.common.initializer.Zero()()
    elif classname.find('BatchNorm') != -1:
        mindspore.common.initializer.Normal()(0.01, 1.0)
        mindspore.common.initializer.Zero()()
    elif classname.find('Linear') != -1:
# torch.nn.init.xavier_normal_
        mindspore.common.initializer.Zero()()
