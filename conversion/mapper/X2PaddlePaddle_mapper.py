"""
迁移程序关于PaddlePaddle的辅助映射类
"""


class Mapper:
    def __init__(self, source_framework: str, para_dict: dict):
        self.source_framework = source_framework
        self.para_dict = para_dict
        self.mapping_function_dict = {
            "paddle.nn.Sequential": "paddle_nn_Sequential",
            "paddle.nn.Conv2D": "paddle_nn_Conv2D",
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

    def paddle_nn_Sequential(self):
        """
        如果目标算子名称是 paddle_nn_Sequential，则将变量的处理推迟到 以后
        :return:
        """
        res = None
        return False, res

    def paddle_nn_Conv2D(self):
        special_para = {}
        if self.source_framework == 'PyTorch':
            bias = self.para_dict['bias']['value']

            if isinstance(bias, bool) and bias is False:
                special_para['bias_attr'] = False

        return True, special_para