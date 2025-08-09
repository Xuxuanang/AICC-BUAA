import ast
import os.path
from okg.scripts import match
import conversion.mapper.X2PyTorch_mapper as X2PyTorch_mapper
import conversion.mapper.X2MindSpore_mapper as X2MindSpore_mapper
import conversion.mapper.X2PaddlePaddle_mapper as X2PaddlePaddle_mapper
from decimal import Decimal
import re

class ASTTransformer(ast.NodeTransformer):
    """
    这是不同深度学习框架下神经网络代码的自动迁移类
    """
    def __init__(self, _source_framework, _target_framework, _source_version='1.5.0', _target_version='1.5.0'):
        # 源框架名称
        self.source_framework = _source_framework
        # 目标框架名称
        self.target_framework = _target_framework
        # 源框架版本
        self.source_version = _source_version
        # 目标框架版本
        self.target_version = _target_version

        # 源ast节点
        self.source_node = None

        self.okg = match.OKG()  # 算子知识图谱
        self.udi = self.UDepInfo()  # Universal dependency information 通用依赖信息列表
        self.soi = []  # Specific operator information list 特定算子信息列表
        self.cur_assign_target = None  # 当前赋值语句的赋值目标
        self.custom_operator_flag = False  # 是否正在遍历自定义算子的标记语句，为了target服务
        self.meta_mapper = self.get_meta_mapper()

        # Custom Operator Map 自定义算子映射关系
        self.com = {}  # 自定义算子函数名称 -> CustomOperatorNode

        self.succeed_api_num=0  #成功转换的算子数量
        self.failed_api_num=0  #转换失败的算子数量

        if self.source_framework == 'PyTorch':
            self.source_prefix = "torch.*"
        elif self.source_framework == 'MindSpore':
            self.source_prefix = "mindspore.*"
        elif self.source_framework == 'PaddlePaddle':
            self.source_prefix = "paddle.*"
    
    class UDepInfo:  # Universal dependency information 通用依赖信息
        def __init__(self):
            self.udi_dict = {}

        def add(self, from_info, import_info, as_info=''):
            udi_key = ''
            if as_info != '':
                udi_key = as_info
            else:
                udi_key = import_info
            full_name = ''
            if from_info == '':
                full_name = import_info
            else:
                full_name = from_info + '.' + import_info
            udi_value = {'full_name': full_name, 'from': from_info, 'import': import_info, 'as': as_info}
            self.udi_dict[udi_key] = udi_value

    class SOpInfo:  # Specific operator information 特定算子信息
        def __init__(self, var):
            self.var = var
            self.source_name = ''
            self.target_name = ''
            self.lineno = ''
            self.col_offset = ''
            self.end_lineno = ''
            self.end_col_offset = ''

        def set_source_name(self, source_name):
            self.source_name = source_name

        def set_target_name(self, target_name):
            self.target_name = target_name

    class CustomOperatorNode:  # 自定义算子节点
        def __init__(self, _operator_name, _operator_type, _distinction):
            # 自定义算子的函数名，即type和distinction的组合
            self.operator_name = _operator_name

            # 自定义算子类型，即知识图谱中存储的自定义算子中的name属性（如CustomDataLoader）；可能会有多个同类型的自定义算子，所以需要进一步区分
            self.operator_type = _operator_type

            # 自定义算子区分度，用来区分同类型的自定义算子，即多对多算子中源算子的主体算子名称及其特征（如CIFAR10 及 train_data）
            self.distinction = _distinction

            # 自定义算子的参数形参
            self.parameter_list = []

            # 自定义算子的函数体节点列表 CustomOperatorBody
            self.bodies = []  # 包括序号、目标算子名称、赋值目标变量名称、原始语句

            # 自定义算子的返回值列表
            self.returns = []

            # 自定义算子的调用者节点
            self.custom_operator_caller = self.CustomOperatorCaller()

            # 自定义算子的ast节点
            self.custom_operator_ast_node = None  # def custom()...

        class CustomOperatorBody:  # 自定义算子的函数体节点
            def __init__(self, _id, _assign_target, _operator_name):
                # 第几行的序号
                self.id = _id

                # 赋值目标变量名称
                self.assign_target = _assign_target

                # 目标算子名称
                self.operator_name = _operator_name

                # 目标算子参数字典
                self.parameter_dict = {}

                # 目标算子调用函数字典 字典套字典：{map:{operations:transform}}
                self.function_dict = {}

                # 原始语句
                self.statement = ''

        class CustomOperatorCaller:  # 自定义算子调用者节点
            def __init__(self):
                # 自定义算子调用者的赋值目标变量名称
                self.assign_target_name = ''

                # 自定义算子调用者的算子名称
                self.custom_operator_caller_name = ''

                # 自定义算子调用者的实参列表
                self.parameter_list = []

                # 自定义算子调用者的ast节点
                self.custom_operator_caller_ast_node = None  # target = custom()...

    def get_meta_mapper(self):
        """
        获取目标框架的辅助映射器
        :return:
        """
        meta_mapper = None
        if self.target_framework == 'PyTorch':
            meta_mapper = X2PyTorch_mapper
        elif self.target_framework == 'MindSpore':
            meta_mapper = X2MindSpore_mapper
        elif self.target_framework == 'PaddlePaddle':
            meta_mapper = X2PaddlePaddle_mapper
        return meta_mapper

    def get_operator_full_name(self, operator_name):
        """
        根据算子名称获取其完整名称
        :param func_name:
        :return:
        """
        operator_full_name = ''
        operator_name_list = operator_name.split('.')
        for i in range(len(operator_name_list)):
            if i == 0:
                operator_name = operator_name_list[i]
            else:
                operator_name = operator_full_name + '.' + operator_name_list[i]
            if operator_name in self.udi.udi_dict:
                operator_full_name = self.udi.udi_dict[operator_name]['full_name']  # 将函数调用的名称补充完整
                for j in range(len(operator_name_list) - i - 1):
                    operator_full_name = operator_full_name + '.' + operator_name_list[j + i + 1]
                break
        return operator_full_name

    def generate_import(self):
        tmp_alias = ast.alias()
        if self.target_framework == 'PyTorch':
            tmp_alias.name = 'torch'
        elif self.target_framework == 'MindSpore':
            tmp_alias.name = 'mindspore'
        elif self.target_framework == 'PaddlePaddle':
            tmp_alias.name = 'paddle'
        new_node = ast.Import()
        tmp_names = [tmp_alias]
        new_node.names = tmp_names
        return new_node

    @staticmethod
    def generate_attribute(operator_name):
        """
        根据算子名称生成ast.Attribute：属性访问节点
        class ast.Attribute(value, attr, ctx)
            属性访问，例如 d.keys。 value 是一个节点，通常为 Name。 attr 是一个给出属性名称的纯字符串，而 ctx 根据属性操作的方式可以为 Load, Store 或 Del。
        :param operator_name:
        :return:
        """
        operator_name_list = operator_name.split('.')
        new_node = ast.Name()
        for index in range(len(operator_name_list)):
            name = operator_name_list[index]
            if index == 0:
                new_node.id = name
                new_node.ctx = ast.Load
            else:
                tmp_node = ast.Attribute()
                tmp_node.value = new_node
                tmp_node.attr = name
                tmp_node.ctx = ast.Load
                new_node = tmp_node
        return new_node

    @staticmethod
    def generate_constant(constant):
        """
        根据算子名称生成ast.Constant：常量节点
        :param constant:
        :return:
        """
        new_node = ast.Constant()
        new_node.value = constant
        return new_node

    @staticmethod
    def generate_decimal(value):
        new_node = ast.Constant()
        new_node.value = float(value)
        return new_node

    def generate_tuple(self, tuple):
        """
        根据算子名称生成ast.Tuple：属性访问节点
        :param tuple:
        :return:
        """
        new_node = ast.Tuple()
        tmp_elts = []
        for constant in tuple:
            tmp_elts.append(self.generate_constant(constant))
        new_node.elts = tmp_elts
        new_node.ctx = ast.Load
        return new_node

    def generate_list(self, list):
        """
        根据算子名称生成ast.List：属性访问节点
        :param tuple:
        :return:
        """
        new_node = ast.List()
        tmp_elts = []
        for constant in list:
            tmp_elts.append(self.generate_constant(constant))
        new_node.elts = tmp_elts
        new_node.ctx = ast.Load
        return new_node

    @staticmethod
    def generate_name(name):
        """
        根据算子名称生成ast.Name：变量名
        :param name:
        :return:
        """
        new_node = ast.Name()
        new_node.id = name
        return new_node

    @staticmethod
    def generate_bool(value):
        new_node = ast.BoolOp()
        return new_node

    def generate_call(self, func, args, keywords: list):
        test = func + '(' + '' + keywords[0] + '='
        if isinstance(keywords[1], int):
            test = test + str(keywords[1]) + ')'
        elif isinstance(keywords[1], str):
            test = test + keywords[1][1:] + ')'
        elif isinstance(keywords[1], bool):
            test = test + str(keywords[1]) + ')'
        return ast.parse(test)

    def generate_call_callback(self, func, args, keywords: list):
        test = func + '(' + '' + keywords[0] + '=' + keywords[1] + ")"
        return ast.parse(test)

    @staticmethod
    def handle_args(operator_args):
        """
        生成算子的args列表
        :param operator_args:
        :return:
        """
        source_args_list = []
        for arg in operator_args:
            if isinstance(arg, ast.Constant):  # 常量
                if isinstance(arg.value, str):
                    source_args_list.append('0' + arg.value)  # 前置0表示这个字符串是一个字面量
                else:
                    source_args_list.append(arg.value)
            elif isinstance(arg, ast.Tuple):
                tuple_tmp = []
                for constant_node in arg.elts:
                    if isinstance(constant_node,ast.Name):
                        tuple_tmp.append('1' + constant_node.id)
                    elif isinstance(constant_node,ast.Call):
                        tuple_tmp.append(ast.unparse(constant_node))
                    else:
                        tuple_tmp.append(constant_node.value)
                source_args_list.append(tuple(tuple_tmp))
            elif isinstance(arg, ast.Name):  # 变量名
                source_args_list.append('1' + arg.id)  # 前置1表示这个字符串是一个变量
            elif isinstance(arg, ast.Attribute):
                source_args_list.append('1' + ast.unparse(arg))
            elif isinstance(arg, ast.BinOp):  # 双目运算
                source_args_list.append('1' + ast.unparse(arg))
            elif isinstance(arg, ast.Subscript):  # 抽取操作，如 l[1]
                source_args_list.append('1' + ast.unparse(arg))
        return source_args_list

    @staticmethod
    def handle_keywords(operator_keywords: list):
        """
        生成算子的keywords字典
        :param operator_keywords:
        :return:
        """
        source_keywords_list = {}
        for keyword in operator_keywords:
            keyword_arg = keyword.arg
            keyword_value = keyword.value
            if isinstance(keyword_value, ast.Constant):
                if isinstance(keyword_value.value, str):
                    source_keywords_list[keyword_arg] = '0' + keyword_value.value
                else:
                    source_keywords_list[keyword_arg] = keyword_value.value
            elif isinstance(keyword_value, ast.Name):
                source_keywords_list[keyword_arg] = '1' + ast.unparse(keyword_value)
            elif isinstance(keyword_value, ast.Subscript):
                source_keywords_list[keyword_arg] = '1' + ast.unparse(keyword_value)
            elif isinstance(keyword_value, ast.Tuple):
                tuple_tmp = []
                for constant_node in keyword_value.elts:
                    tuple_tmp.append(constant_node.value)
                source_keywords_list[keyword_arg] = tuple(tuple_tmp)
            elif isinstance(keyword_value, ast.List):
                tuple_tmp = []
                for constant_node in keyword_value.elts:
                    print(constant_node)
                    if isinstance(constant_node,ast.Name):
                        tuple_tmp.append(constant_node.id)
                    else:
                        # print(type(constant_node))
                        tuple_tmp.append(constant_node.value)
                source_keywords_list[keyword_arg] = tuple_tmp
            elif isinstance(keyword_value, ast.BinOp):  # 双目运算
                source_keywords_list[keyword_arg] = '1' + ast.unparse(keyword_value)
            elif isinstance(keyword_value, ast.Attribute):
                source_keywords_list[keyword_arg] = '1' + ast.unparse(keyword_value)
        return source_keywords_list

    @staticmethod
    def supplement_parameter(source_operator_para_dict: dict, source_args_list: list,
                             source_keywords_dict: dict) -> dict:
        """
        更新算子的keywords列表
        :param source_operator_para_dict:
        :param source_args_list:
        :param source_keywords_dict:
        :return:
        """
        idx = 0
        for para, para_info in source_operator_para_dict.items():
            if idx < source_args_list.__len__():
                para_info['value'] = source_args_list[idx]
            elif para in source_keywords_dict:
                para_info['value'] = source_keywords_dict[para]
            else:
                para_info['value'] = 'Null'
            # if isinstance(para_info['value'], str) and para_info['value'] != 'Null':
            #     para_info['value'] = '0' + para_info['value']
            if para_info['value'] == 'True':
                para_info['value'] = True
            elif para_info['value'] == 'False':
                para_info['value'] = False
            idx += 1
        return source_operator_para_dict

    def visit_Import(self, node: ast.Import):
        """
        构建通用依赖信息
        :param node:
        :return:
        """
        for alias in node.names:
            as_info = ''
            if alias.asname:
                as_info = alias.asname
            self.udi.add(from_info='', import_info=alias.name, as_info=as_info)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
        构建通用依赖信息
        :param node:
        :return:
        """
        for alias in node.names:
            as_info = ''
            if alias.asname:
                as_info = alias.asname
            self.udi.add(from_info=node.module, import_info=alias.name, as_info=as_info)
        return node

    def get_custom_operator_name(self, source_name, source_id, source_para: dict):
        """
        获取source_name算子对应的目标框架下自定义算子的函数名
        自定义算子的函数名需要根据 知识图谱中存储的自定义算子节点的name属性值 以及 当前算子的distinction属性值来进行拼接
        :param source_name:
        :param source_id:
        :param source_para:
        :return:
        """
        # 获取source_name算子对应目标框架的自定义算子的name属性值
        custom_operator_type = self.okg.get_custom_operator_name(source_name, self.target_framework)
        # 获取source_id算子的distinction属性值
        distinction = self.okg.get_distinction(source_id)

        custom_operator_name = custom_operator_type
        if distinction.find("parameter_") != -1:
            # 说明自定义算子需要通过参数来进一步区分
            distinction_value = source_para.get(distinction[10:]).get('value')
            if isinstance(distinction_value, str) and distinction_value[0] == '1':
                distinction_value = distinction_value[1:]
            custom_operator_name = custom_operator_name + '_' + distinction_value
        elif distinction == "assign":
            # 说明自定义算子需要通过赋值目标的变量名称来进一步区分
            custom_operator_name = custom_operator_name + '_' + self.cur_assign_target
        return custom_operator_name

    def get_custom_operator(self, source_name, source_id, source_para: dict) -> CustomOperatorNode:
        """
        根据 CustomOperatorRelationship 获取对应的自定义算子节点 CustomOperatorNode
        :param source_name:
        :param source_id:
        :param source_para:
        :return:
        """
        # 获取source_name算子对应的目标框架下自定义算子的函数名
        cur_custom_operator_name = self.get_custom_operator_name(source_name, source_id, source_para)

        # 自定义算子映射关系 dict
        custom_operator_map = self.com

        # 映射关系中不存在该自定义算子
        if cur_custom_operator_name not in custom_operator_map:
            # 获取source_name算子对应目标框架的自定义算子的name属性值
            custom_operator_type = self.okg.get_custom_operator_name(source_name, self.target_framework)
            # 获取source_id算子的distinction属性值
            distinction = self.okg.get_distinction(source_id)
            distinction_value = ''
            # 生成自定义算子节点
            if distinction.find("parameter_") != -1:
                # 说明自定义算子需要通过参数来进一步区分
                distinction_value = source_para.get(distinction[10:]).get('value')
                if isinstance(distinction_value, str) and distinction_value[0] == '1':
                    distinction_value = distinction_value[1:]
            elif distinction == "assign":
                # 说明自定义算子需要通过赋值目标的变量名称来进一步区分
                distinction_value = self.cur_assign_target
            custom_operator = self.CustomOperatorNode(cur_custom_operator_name, custom_operator_type, distinction_value)
            # 将映射关系存储到map中
            custom_operator_map[cur_custom_operator_name] = custom_operator

        return custom_operator_map.get(cur_custom_operator_name)

    def get_custom_operator_body(self, custom_operator: CustomOperatorNode,
                                 operator_name) -> CustomOperatorNode.CustomOperatorBody:
        """
        获取自定义算子custom_operator的函数体列表中operator_name对应的CustomOperatorBody
        :param custom_operator:
        :param operator_name:
        :return:
        """
        # 获取自定义算子的函数体列表
        bodies = custom_operator.bodies

        idx = 1
        for i, body in enumerate(bodies):
            idx = idx + 1
            if body.operator_name == operator_name:
                return body

        new_body = self.CustomOperatorNode.CustomOperatorBody(idx, self.cur_assign_target, operator_name)
        bodies.append(new_body)
        return new_body

    @staticmethod
    def get_custom_operator_returns(custom_operator_node: CustomOperatorNode):
        """

        :param custom_operator_node:
        :return:
        """
        # 自定义算子的返回值列表
        returns = []
        # 自定义算子的函数体列表
        bodies = custom_operator_node.bodies
        for body in bodies:
            if body.assign_target is not None:
                returns.append(body.assign_target)
        custom_operator_node.returns = returns

    def generate_custom_operator(self, custom_operator_node: CustomOperatorNode):
        """

        :param custom_operator_node:
        :return:
        """
        # 自定义算子函数头中的函数名
        operator_name = custom_operator_node.operator_name
        # 自定义算子函数头中的参数列表
        parameter_list = custom_operator_node.parameter_list
        # 自定义算子的函数体列表
        bodies = custom_operator_node.bodies

        # 加入自定义算子的函数头
        res = 'def ' + operator_name + '('
        for parameter in parameter_list:
            res = res + parameter + ','
        if len(parameter_list) != 0:
            res = res[:-1]
        res = res + '):\n\t'

        # 加入自定义算子的函数体
        bodies_ast = self.generate_custom_operator_bodies(bodies)
        for body_ast in bodies_ast:
            res = res + ast.unparse(body_ast) + '\n\t'

        # 加入return语句
        self.get_custom_operator_returns(custom_operator_node)
        if len(custom_operator_node.returns) != 0:
            res = res + 'return '
            for custom_operator_node_return in custom_operator_node.returns:
                res = res + custom_operator_node_return + ','
            res = res[:-1] + '\n\t'

        return ast.parse(res)

    def generate_custom_operator_bodies(self, bodies: list) -> list:
        """
        生成自定义算子中的body体，其中res包含多个body语句，每个body语句都是一个ast节点，最终组成一个ast节点列表
        :param bodies:
        :return:
        """
        res = []
        for body in bodies:
            res.append(self.generate_custom_operator_body(body))
        return res

    def generate_custom_operator_body(self, custom_operator_body: CustomOperatorNode.CustomOperatorBody) -> ast.AST:
        """
        生成自定义算子中的单个body的ast节点
        :param custom_operator_body:
        :return:
        """
        # 赋值目标的变量名称
        assign_target = custom_operator_body.assign_target
        # 目标框架下算子名称
        operator_name = custom_operator_body.operator_name
        # 算子的参数列表
        parameter_dict = custom_operator_body.parameter_dict
        # 算子的调用函数列表
        function_dict = custom_operator_body.function_dict
        # body的原始语句
        statement = custom_operator_body.statement

        # 加入赋值目标的变量名称
        if assign_target is not None:
            statement = assign_target + ' = '

        # 加入算子名称和参数列表
        statement = statement + ast.unparse(self.generate_Call(operator_name, [], parameter_dict))

        # 加入算子的调用函数列表
        if len(function_dict) != 0:
            for name in function_dict:
                statement = statement + '.' + ast.unparse(self.generate_Call(name, [], function_dict.get(name)))
        return ast.parse(statement)

    def generate_Call(self, operator_name: str, args: list, keywords: dict) -> ast.AST:
        """
        生成Call的ast节点
        :param operator_name:
        :param args:
        :param keywords:
        :return:
        """
        # 加入算子名称
        statement = operator_name
        statement = statement + '('
        # 加入算子的args
        for arg in args:
            statement = statement + self.get_value(arg) + ','
        # 加入算子的keywords
        for key in keywords:
            statement = statement + key + '=' + self.get_value(keywords.get(key)) + ','
        if statement[:-1] == ',':
            statement = statement[:-1] + ')'
        else:
            statement = statement + ')'
        return ast.parse(statement)

    @staticmethod
    def get_value(value) -> str:
        """
        获取value在语句中真实的字符串（因为可能存在一个字符串既可能字面量，也可能是变量名）
        :param value:
        :return:
        """
        res = ''
        if isinstance(value, int):
            res = str(value)
        elif isinstance(value, str):
            if value[0] == '0':  # 字面量
                res = '\'' + value[1:] + '\''
            else:  # 变量名
                res = value[1:]
        elif isinstance(value, bool):
            res = str(value)
        return res

    @staticmethod
    def generate_custom_operator_caller(custom_operator_caller: CustomOperatorNode.CustomOperatorCaller) -> ast.AST:
        """
        生成自定义算子调用者的ast节点
        :param custom_operator_caller:
        :return:
        """
        if custom_operator_caller.custom_operator_caller_ast_node is not None:
            # 将之前的调用语句置空，存储的调用语句为ast.Module，所以将body清楚可以达到置空的效果
            custom_operator_caller.custom_operator_caller_ast_node.body.clear()

        assign_target_name = custom_operator_caller.assign_target_name
        caller_name = custom_operator_caller.custom_operator_caller_name
        parameter_list = custom_operator_caller.parameter_list
        ast_node = custom_operator_caller.custom_operator_caller_ast_node

        statement = ''
        if assign_target_name != '':
            statement = statement + assign_target_name + ' = '
        statement = statement + caller_name + '('
        if len(parameter_list) != 0:
            for parameter in parameter_list:
                statement = statement + parameter + ','
            statement = statement[:-1]
        statement = statement + ')'
        return ast.parse(statement)

    def visit_Call(self, node: ast.Call):
        """
        遍历Call语句，即函数调用语句
        :param node:
            ast.Call(func, args, keywords, starargs, kwargs)
            一个函数调用。 func 是函数，它通常是一个 Name 或 Attribute 对象。 对于其参数:
                args 保存由按位置传入的参数组成的列表。
                keywords 保存了一个代表以关键字传入的参数的 keyword 对象的列表。
            当创建一个 Call 节点时，需要有 args 和 keywords，但它们可以为空列表。 starargs 和 kwargs 是可选的。
        :return:
        """
        super(ASTTransformer, self).generic_visit(node)
        # 目标框架下的新ast节点
        target_node = None
        # if (isinstance(node.func, ast.Call)):
        #     return self.visit(node.func)
        # 将源算子的名称扩充为完整名称，如 Conv2d 扩充为 torch.nn.Conv2d
        source_name = self.get_operator_full_name(ast.unparse(node.func))
        # 获取当前算子在okg中的id
        source_id = self.okg.get_operator_id(self.source_framework, source_name)
        # 当前算子不需要根据okg做映射
        if source_id == -1:
            if(re.match(self.source_prefix,source_name)):
                # print(source_name)
                self.failed_api_num += 1
            return node
        if(not re.match(self.source_prefix,source_name)):
            return node
        # 多对多算子
        if self.okg.get_mapping(source_id) == "n:n":
            self.custom_operator_flag = True
            target_node = self.visit_Call_custom(node, source_name, source_id)
        # 一对一算子
        else:
            target_node = self.visit_Call_Model(node, source_name, source_id)
        # # 数据处理算子
        # else source_name.find("torchvision.datasets") != -1 or source_name.find("DataLoader") != -1:
        #     target_node = self.visit_datasets(node, source_name, source_id)

        return target_node

    def visit_Call_custom(self, node: ast.Call, source_name, source_id):
        """
        自定义算子的迁移程序
        :param node: 函数调用的ast节点
        :param source_name: 算子名称
        :param source_id: 算子在知识图谱中的节点id
        :return:
        """
        # 1. 首先获取源算子的所有信息
        # 1.1 获取源算子的args列表 list
        source_args = self.handle_args(node.args)
        # 1.2 获取源算子的keywords字典 dict
        source_keywords = self.handle_keywords(node.keywords)

        # 1.3 获取源算子的完整参数字典（仅包含默认值） dict
        source_para, _ = self.okg.get_parameters(source_id)
        # 将source_args 和 source_keywords 的值补充进完整参数字典中（包含自定义值和默认值） dict
        source_para = self.supplement_parameter(source_para, source_args, source_keywords)

        # 1.4 根据反射获取迁移辅助函数 TODO
        mapper = self.meta_mapper.Mapper(self.source_framework, source_para)
        mapper_str = mapper.mapping_function_dict.get(source_name, 'mapper_none')
        mapper_func = getattr(mapper, mapper_str)
        mapper_res_flag, mapper_res = mapper_func()  # 获取应该修改的参数名称和对应的值

        # 2. 其次开始根据源算子生成自定义算子 包括自定义算子的函数名、形参和函数体
        # 2.1 获取自定义算子节点
        custom_operator = self.get_custom_operator(source_name, source_id, source_para)
        custom_operator_parameter_list = custom_operator.parameter_list  # 自定义算子的形参列表

        # 2.2 生成自定义算子的函数体，需要根据源算子及其各个参数来决定生成新的目标算子还是在原有目标算子中修改
        # 遍历源算子的完整参数字典
        for parameter in source_para.values():
            # 获取当前参数的名称
            parameter_name = parameter.get('name')

            # 获取当前参数的知识图谱节点
            source_parameter_id = self.okg.get_parameter_id(source_id, parameter_name)
            # 若不存在对应关系，则跳过
            if source_parameter_id == -1:
                continue

            # 获取当前参数的参数值
            parameter_value = parameter.get('value')
            # 获取当前参数的默认值
            parameter_default = parameter.get('default')
            # 获取目标框架下对应的参数节点id
            target_parameter_id = self.okg.get_equivalent_parameter_id(source_parameter_id, self.target_framework)
            # 若不存在对应关系，则跳过
            if target_parameter_id == -1:
                continue

            # 获取目标框架下对应的参数名称
            target_parameter_name = self.okg.get_name(target_parameter_id)
            # 获取目标框架下对应的迁移类型，如parameter2parameter、parameter2function、function2parameter
            equivalent_parameter_type = self.okg.get_equivalent_parameter_type(source_parameter_id, target_parameter_id,
                                                                               self.target_framework)
            # 获取目标框架下对应的算子id
            target_id = self.okg.get_target_id(source_parameter_id, self.target_framework)
            # 获取目标框架下对应的算子名称
            target_name = self.okg.get_operator_full_name(target_id)

            cur_body = self.get_custom_operator_body(custom_operator, target_name)
            if self.cur_assign_target is not None:
                cur_body.assign_target = self.cur_assign_target

            if parameter_value == 'Null' and parameter_default is None:  # 表明源算子中没有使用该参数 并且该参数没有默认值
                continue  # 不进行迁移，跳过
            elif parameter_value == 'Null' and parameter_default is not None:  # 表明源算子中没有使用该参数 但该参数有默认值

                if equivalent_parameter_type[-9:] == 'parameter':  # parameter2parameter 或者 function2parameter
                    cur_body.parameter_dict[target_parameter_name] = parameter_default
                elif equivalent_parameter_type[-8:] == 'function':
                    function_name = self.okg.get_function_of_parameter(target_parameter_id, target_parameter_name)
                    cur_body.function_dict[function_name] = {target_parameter_name: parameter_default}
            else:  # 表明源算子中使用了该参数
                # 将目标算子的参数名和当前参数值进行匹配
                caller = custom_operator.custom_operator_caller
                if equivalent_parameter_type[-9:] == 'parameter':  # parameter2parameter 或者 function2parameter
                    cur_body.parameter_dict[target_parameter_name] = parameter_value
                    # # 若当前参数值是变量名，则暂存到自定义算子的形参列表中
                    # custom_operator_parameter_list.append(target_parameter_name)
                    if isinstance(parameter_value, str) and parameter_value[0] == '1':
                        custom_operator_parameter_list.append(parameter_value[1:])
                    # 若当前参数值是变量名，还要暂存到自定义算子调用者的实参列表中
                    if isinstance(parameter_value, str) and parameter_value[0] == '1':
                        caller.parameter_list.append(parameter_value[1:])
                elif equivalent_parameter_type[-8:] == 'function':
                    function_name = self.okg.get_function_of_parameter(target_parameter_id, target_parameter_name)
                    cur_body.function_dict[function_name] = {target_parameter_name: parameter_value}
                    # 若当前参数值是变量名，则暂存到自定义算子的形参列表中
                    if isinstance(parameter_value, str) and parameter_value[0] == '1':
                        # custom_operator_parameter_list.append(function_name + '_' + target_parameter_name)
                        custom_operator_parameter_list.append(parameter_value[1:])
                    # 若当前参数值是变量名，还要暂存到自定义算子调用者的实参列表中
                    if isinstance(parameter_value, str) and parameter_value[0] == '1':
                        caller.parameter_list.append(parameter_value[1:])

        # 3. 根据自定义算子类生成相应的自定义算子的ast节点
        custom_operator.custom_operator_ast_node = self.generate_custom_operator(custom_operator)

        # 4. 生成自定义算子调用者的ast节点，并返回该节点
        # 获取自定义算子调用节点
        caller = custom_operator.custom_operator_caller
        # 将当前赋值目标变量值加入到自定义算子调用节点赋值目标变量列表中
        if self.cur_assign_target is not None:
            caller.assign_target_name = self.cur_assign_target
        caller.custom_operator_caller_name = custom_operator.operator_name
        caller.custom_operator_caller_ast_node = self.generate_custom_operator_caller(caller)

        return caller.custom_operator_caller_ast_node

    def visit_Call_Model(self, node: ast.Call, source_name, source_id):
        """
        一对一算子的迁移程序
        :param node:
        :param source_name: 源算子的完整名称
        :param source_id: 当前算子在okg中的id
        :return:
        """
        # 0. 目标框架下的新ast节点
        target_node = node
        # print(source_name,source_id)
        # 1. 首先获取源算子的所有信息
        # 1.1 获取源算子的args列表 list
        source_args = self.handle_args(node.args)
        # 1.2 获取源算子的keywords字典 dict
        source_keywords = self.handle_keywords(node.keywords)
        # 1.3 获取源算子的完整参数信息
        # 获取源算子的完整参数字典 (仅包含默认值) dict
        source_para, _ = self.okg.get_parameters(source_id)
        # 将source_args 和 source_keywords 的值补充进完整参数字典中(包含自定义值和默认值) dict
        source_para = self.supplement_parameter(source_para, source_args, source_keywords)

        # print(source_para)
        # 2. 其次开始根据源算子生成目标算子
        # 2.0 获取目标算子在okg中的id
        target_id = self.okg.get_equivalent_operator_id(source_id, self.target_framework)
        if(target_id == -1):
            self.failed_api_num += 1
            return node
        
        self.succeed_api_num += 1

        # 2.1 修改新ast节点的算子名称
        # 获取目标算子名称
        target_operator_full_name = self.okg.get_operator_full_name(target_id)
        # 根据算子名称生成节点
        target_operator_func = self.generate_attribute(target_operator_full_name)
        # 修改目标ast节点
        target_node.func = target_operator_func

        # 2.2 修改新ast节点的算子参数（或input）
        # 由于可能存在源算子与目标算子参数用法不一致的问题（如所支持的参数类型不一致、参数含义不一致），因此需要根据反射获取辅助映射函数
        # 获取辅助映射对象
        mapper = self.meta_mapper.Mapper(self.source_framework, source_para)
        # 获取辅助映射函数名称，如果不存在，则使用默认值
        mapping_func_name = mapper.mapping_function_dict.get(target_operator_full_name, 'mapper_none')
        # 根据反射获取辅助映射类
        mapping_func = getattr(mapper, mapping_func_name)
        # 根据辅助映射函数生成：是否跳过映射 以及 目标算子的特殊参数值
        is_mapping, special_para = mapping_func()

        # 2.2.1 不需要迁移，则进一步遍历算子中的参数
        if not is_mapping:
            for index, old_arg in enumerate(node.args):
                if isinstance(old_arg, list):
                    new_arg = []
                    for arg in old_arg:
                        if isinstance(arg, ast.AST):
                            arg = self.visit(arg)
                            if arg is None:
                                continue
                            elif not isinstance(arg, ast.AST):
                                new_arg.extend(arg)  # 在列表末尾一次性追加另一个序列中的多个值
                                continue
                        new_arg.append(arg)  # 将迁移后的value追加到new_values列表末尾
                    old_arg[:] = new_arg
                elif isinstance(old_arg, ast.AST):
                    self.visit(old_arg)
            return target_node

        # 2.2.2 需要迁移，则遍历获取目标框架下算子的各个参数值以及input值
        # 获取目标算子参数名称列表和input名称列表
        _, target_para = self.okg.get_parameters(target_id)
        _, target_input = self.okg.get_inputs(target_id)
        # 目标算子的args列表
        target_args = []
        # 目标算子的keywords列表
        target_keywords = []
        # 目标算子的inputs列表
        target_inputs = []
        # 分割args和keywords的标志，其中false代表args，True代表keywords
        div_flag = False

        # 遍历目标算子的参数列表
        for parameter in target_para:
            # 2.2.2.1 获取目标算子中当前参数对应在源框架中的参数id
            # 获取目标算子的参数节点id
            target_para_id = self.okg.get_parameter_id(target_id, parameter)
            # print(target_para_id)
            # 获取源算子的对应参数节点id
            source_para_id = self.okg.get_equivalent_parameter_id(target_para_id, self.source_framework)
            
            # 2.2.2.2 获取源框架中对应参数的参数值
            if source_para_id == -1 and parameter not in special_para:  # 如果不存在，则跳过
                continue
            elif source_para_id == -1:
                source_para_value = special_para[parameter]
            else:
                # 获取源算子的参数属性
                source_para_info = self.okg.get_parameter_info(source_para_id)
                # 获取源算子的参数名称
                source_para_name = source_para_info['name']

                # print(source_name,source_para_name,source_para_id)
                if source_para_name == "-":
                    continue
                else:
                    # 获取源算子的参数值
                    source_para_value = source_para[source_para_name]['value']



            # 2.2.2.3
            # 操作args 
            if source_para_value != 'Null' and div_flag is False:
                if parameter in special_para:
                    source_para_value = special_para.get(parameter)
                target_args.append({parameter: source_para_value})
            # 操作keywords
            elif source_para_value != 'Null' and div_flag is True:
                if parameter in special_para:
                    source_para_value = special_para.get(parameter)
                target_keywords.append({parameter: source_para_value})
            # args 和 keywords的分界点
            elif source_para_value == 'Null':  # 因为源算子的该参数没有该值 所以目标算子中该参数就空出 因此后续就需要转为keywords
                div_flag = True
                if parameter in special_para:
                    source_para_value = special_para.get(parameter)
                    target_keywords.append({parameter: source_para_value})
            # 当前参数在辅助映射函数生成的特殊参数中
            elif parameter in special_para:
                source_para_value = special_para.get(parameter)
                target_keywords.append({parameter: source_para_value})
            else:
                continue

        # 遍历目标算子的input列表 TODO
        for input in target_input:
            # 获取目标算子的参数节点id
            target_input_id = self.okg.get_input_id(target_id, input)
            # 获取源算子的对应参数节点id
            source_para_id = self.okg.get_equivalent_parameter_id(target_input_id, self.source_framework)

            # 如果不存在，则跳过
            if source_para_id == -1:
                continue

            # 获取源算子的参数属性
            source_para_info = self.okg.get_parameter_info(source_para_id)
            # 获取源算子的参数名称
            source_para_name = source_para_info['name']
            source_input_value = None
            # 获取源算子的参数值
            if source_para_name in source_para:
                source_input_value = source_para[source_para_name]['value']

            # 操作inputs
            if input in special_para:
                source_input_value = special_para.get(input)

            if source_input_value is not None:
                target_inputs.append(source_input_value)

        # 2.2.3 利用上一步获取的参数值生成对应的ast节点，并拼接到target_node中
        # 目标算子的args节点列表
        target_node_args = []
        # 目标算子的keywords节点列表
        target_node_keywords = []
        # 添加目标算子的args节点
        for arg in target_args:
            for value in arg.values():
                tmp_node = ast.AST()
                if isinstance(value, int):
                    tmp_node = self.generate_constant(value)
                elif isinstance(value, float):
                    tmp_node = self.generate_constant(value)
                elif isinstance(value, Decimal):
                    tmp_node = self.generate_constant(value)
                elif isinstance(value, tuple):
                    tmp_node = self.generate_tuple(value)
                elif isinstance(value, list):
                    tmp_node = self.generate_list(value)
                elif isinstance(value, str):
                    if value[0] == '0':  # 该字符串是一个字面量
                        tmp_node = self.generate_constant(value[1:])
                    elif value[0] == '1':  # 该字符串是一个变量
                        tmp_node = self.generate_name(value[1:])
                elif isinstance(value, bool):
                    tmp_node = self.generate_bool(value)
                elif value is None:
                    tmp_node = self.generate_constant(value)
                target_node_args.append(tmp_node)
        # 添加目标算子的keywords节点
        for keyword in target_keywords:
            for item in keyword.items():
                para = item[0]
                value = item[1]
                tmp_node = ast.AST()
                if isinstance(value, int):
                    tmp_node = self.generate_constant(value)
                elif isinstance(value, tuple):
                    tmp_node = self.generate_tuple(value)
                elif isinstance(value, list):
                    tmp_node = self.generate_list(value)
                elif isinstance(value, str):
                    if value[0] == '0':  # 该字符串是一个字面量
                        tmp_node = self.generate_constant(value[1:])
                    elif value[0] == '1':  # 该字符串是一个变量
                        tmp_node = self.generate_name(value[1:])
                elif isinstance(value, Decimal):
                    tmp_node = self.generate_decimal(value)
                elif isinstance(value, float):
                    tmp_node = self.generate_decimal(value)
                elif value is None:
                    tmp_node = self.generate_constant(value)
                tmp_keyword = ast.keyword()
                tmp_keyword.arg = para
                tmp_keyword.value = tmp_node
                target_node_keywords.append(tmp_keyword)
        # 将args和keywords拼接到target_node中
        target_node.args = target_node_args
        target_node.keywords = target_node_keywords

        # 添加目标算子的input
        target_node_str = ast.unparse(target_node)
        if len(target_inputs) != 0:
            target_node_str = target_node_str + '('
            for value in target_inputs:
                if isinstance(value, str):
                    if value[0] == '0':  # 该字符串是一个字面量
                        target_node_str = target_node_str + '\'' + value[1:] + '\''
                    elif value[0] == '1':  # 该字符串是一个变量
                        target_node_str = target_node_str + value[1:]
                else:
                    target_node_str = target_node_str + value
                target_node_str = target_node_str + ','
            target_node_str = target_node_str[:-1]
            target_node_str = target_node_str + ')'
            return ast.parse(target_node_str)

        return target_node

    def visit_List(self, node: ast.List):
        for test in node.elts:
            if isinstance(test, ast.Call):
                self.visit(test)
                self.custom_operator_flag = False
        return node

    def visit_Assign(self, node: ast.Assign):
        """
        遍历Assign语句，即赋值语句
        :param node:
            class ast.Assign(targets, value, type_comment)
                一次赋值。 targets 是一个由节点组成的列表，而 value 是一个单独节点。
                targets 中有多个节点表示将同一个值赋给多个目标。 解包操作是通过在 targets 中放入一个 Tuple 或 List 来表示的。
        :return:
        """
        # TODO 目前只考虑了单目标赋值情况
        # 将当前赋值目标赋给cur_assign_target
        self.cur_assign_target = ast.unparse(node.targets[0])
        target_node = node
        new_node_value = self.visit(node.value)  # 处理后的赋值ast的value

        if new_node_value is None:
            self.cur_assign_target = None  # 置位
            return None

        if self.custom_operator_flag:  # 如果当前处理的node.value是多对多算子，则直接将当前的赋值语句替换成新的调用者语句
            self.cur_assign_target = None  # 置位
            self.custom_operator_flag = False
            return new_node_value
        else:  # 否则更新node的value
            if isinstance(new_node_value, ast.Module):
                target_node.value = new_node_value.body[0].value
            else:
                target_node.value = new_node_value

        self.cur_assign_target = None  # 置位
        return target_node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        target_node = node

        # 获取函数名称节点
        source_func_name = node.name
        source_func_node_id = self.okg.get_member_function_id(self.source_framework, source_func_name)
        if source_func_node_id != -1:  # 当前函数需要根据okg做映射
            target_func_info = self.okg.get_target_node_info(source_func_node_id, 'member_function',
                                                             self.target_framework,
                                                             'version', 1)
            target_func_name = target_func_info['name']
            target_node.name = target_func_name

        for test in node.body:
            self.visit(test)
        return target_node

    def visit_Attribute(self, node: ast.Attribute):
        """
        处理【属性访问】节点
        【属性访问】:例如 d.keys。 value 是一个节点，通常为 Name。 attr 是一个给出属性名称的纯字符串，而 ctx 根据属性操作的方式可以为 Load, Store 或 Del。
        print(ast.dump(ast.parse('snake.colour', mode='eval'), indent=4))
        Expression(
            body=Attribute(
                value=Name(id='snake', ctx=Load()),
                attr='colour',
                ctx=Load()))
        :param node:
        :return:
        """
        target_node = node

        # 获取属性访问节点的原始字符串
        source_attribute_name = ast.unparse(node)

        # 将源函数调用的名称扩展为完整名称
        source_attribute_full_name = self.get_operator_full_name(source_attribute_name)
        # 获取当前函数在okg中的id
        source_operator_id = self.okg.get_operator_id(self.source_framework,
                                                      source_attribute_full_name)  # 获取当前属性访问在okg中的id

        if(not re.match(self.source_prefix,source_attribute_full_name)):
            return node
        self.failed_api_num += 1
        if source_operator_id != -1:  # 当前函数需要根据okg做映射
            # 获取目标函数在okg中的id
            target_operator_id = self.okg.get_target_id_by_source_id_and_rel(
                source_operator_id,
                'equivalentOperator',
                {'framework_to': self.target_framework}
            )
            if target_operator_id != -1:
                # 修改算子名称
                self.succeed_api_num += 1
                self.failed_api_num -= 1
                target_operator_full_name = self.okg.get_operator_full_name(target_operator_id)  # 获取目标算子名称
                target_node = self.generate_attribute(target_operator_full_name)  # 根据算子名称生成节点
        return target_node

    def visit_Module(self, node):
        """
        遍历抽象语法树的起始处
        :param node: 源代码的抽象语法树
        :return:
        """
        # 将源ast节点存储起来
        self.source_node = node

        # 遍历抽象语法树node并进行迁移
        for field, old_value in ast.iter_fields(node):
            # print(field, old_value)
            if isinstance(old_value, list):
                new_values = []
                if field == 'body':  # 在代码头加入import目标框架的语句
                    new_values.append(self.generate_import())
                for value in old_value:
                    if isinstance(value, ast.AST):
                        # print(value,end=' ')
                        # print(value.__class__.__name__)
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)  # 在列表末尾一次性追加另一个序列中的多个值
                            continue
                    new_values.append(value)  # 将迁移后的value追加到new_values列表末尾
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)

        # 把自定义算子节点加到body首部
        custom_operator_idx = 0
        for body in node.body:
            if isinstance(body, ast.Import) or isinstance(body, ast.ImportFrom):
                custom_operator_idx = custom_operator_idx + 1
        for custom_operator_node in self.com.values():
            node.body.insert(custom_operator_idx, custom_operator_node.custom_operator_ast_node)
        return ast.parse(ast.unparse(node))

    # def visit_ClassDef(self, node: ast.ClassDef):
    #     """
    #     处理【类定义】节点
    #     :param node:
    #     :return:
    #     """
    #     for field, old_value in ast.iter_fields(node):
    #         if field == 'bases':
    #
    #         else:
    #             if isinstance(old_value, list):
    #                 new_values = []
    #                 for value in old_value:
    #                     if isinstance(value, ast.AST):
    #                         value = self.visit(value)
    #                         if value is None:
    #                             continue
    #                         elif not isinstance(value, ast.AST):
    #                             new_values.extend(value)
    #                             continue
    #                     new_values.append(value)
    #                 old_value[:] = new_values
    #             elif isinstance(old_value, ast.AST):
    #                 new_node = self.visit(old_value)
    #                 if new_node is None:
    #                     delattr(node, field)
    #                 else:
    #                     setattr(node, field, new_node)
    #     return node
    def print_info(self):
        print("===========================================\nConvert Summary:\n===========================================")
        # print("There are %d %s APIs in this Project:" %(self.succeed_api_num+self.failed_api_num, self.source_framework))
        # print("%d %s APIS have been converted to %s successfully!" % (self.succeed_api_num, self.source_framework, self.target_framework))
        # print("%d %s APIs are not supported to convert to %s currently!" % (self.failed_api_num, self.source_framework, self.target_framework))
        if (self.succeed_api_num + self.failed_api_num) > 0:
            print("%d %s APIs have been converted to %s successfully!" % (self.succeed_api_num, self.source_framework, self.target_framework))
            print("%d %s APIs failed to converted to %s" % (self.failed_api_num, self.source_framework, self.target_framework))
            print("Convert Rate is: %.2f%%" % (self.succeed_api_num*100 / (self.succeed_api_num + self.failed_api_num)))
        else:
            print("No API recognized")

class ASTProcessor:
    """
    这是在文件和窗口中输入输出源代码、将源代码与ast进行互转等的类
    """

    def __init__(self, _source_path, _target_path):
        # 源代码的路径
        self.__source_path = _source_path
        # 目标代码的路径
        self.__target_path = _target_path
        # 导入源代码并存入source_code
        self.source_code = self.import_source_code(self.__source_path)
        self.ast_root = self.code_to_ast()
        self.target_code = self.ast_to_code()

    @staticmethod
    def import_source_code(pathname) -> str:
        """
        导入pathname处的源代码
        :return:
        """
        with open(pathname, 'r', encoding='utf-8') as f:
            return f.read()

    def write_target_code(self, target_code):
        """
        将目标代码写入文件
        :param target_code:
        :return:
        """
        path = self.__target_path
        paths = os.path.split(path)
        for p in paths:
            if not os.path.exists(p) and p[-3:] != '.py':
                os.makedirs(p)
        with open(path, 'w', encoding='utf-8') as f:
            return f.write(target_code)

    def code_to_ast(self):
        """
        将源代码转换成抽象语法树ast
        :return:
        """
        return ast.parse(source=self.source_code, filename="./log/error.txt")

    def ast_to_code(self) -> str:
        """
        将修改后的抽象语法树转换成目标代码
        :return:
        """
        return ast.unparse(self.ast_root)

    def print_ast(self):
        """
        打印抽象语法树
        :return:
        """
        print(ast.dump(self.ast_root, include_attributes=True, indent="\t"))  # astpretty.pprint(root)

    def ast_to_file(self, filepath):
        """
        将抽象语法树输入到文件中
        """
        with open(filepath, 'w') as file:
            file.write(ast.dump(self.ast_root, include_attributes=True, indent="\t"))

    def print_target_code(self):
        """
        打印目标代码
        :return:
        """
        print(self.target_code)
