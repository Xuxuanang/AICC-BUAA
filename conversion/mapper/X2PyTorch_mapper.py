"""
迁移程序关于PyTorch的辅助映射类
"""


class Mapper:
    def __init__(self, source_framework: str, para_dict: dict):
        self.source_framework = source_framework
        self.para_dict = para_dict
        self.mapping_function_dict = {
            "torch.nn.Sequential": "torch_nn_Sequential",
            "torch.nn.Conv2d": "torch_nn_Conv2d"
        }  # 利用反射找到对应的辅助映射函数

    """
    True代表需要迁移、False代表跳过迁移
    """

    def mapper_none(self):
        """
        :return:
        """
        special_para = {}
        return True, special_para

    def torch_nn_Sequential(self):
        """
        如果目标算子名称是 torch.nn.Sequential，则将变量的处理推迟到 以后
        :return:
        """
        res = None
        return False, res

    def torch_nn_Conv2d(self):
        special_para = {}
        if self.source_framework == 'PaddlePaddle':
            bias_attr = self.para_dict['bias_attr']['value']

            if isinstance(bias_attr, bool) and bias_attr is False:
                special_para['bias'] = False
        special_para['padding_mode'] = '0zeros'
        return True, special_para
