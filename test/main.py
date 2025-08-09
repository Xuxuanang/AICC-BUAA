import os
import shutil
import sys
import re
sys.path.insert(0, sys.path[0]+"/../")
from conversion import transformers


def transform_file(source_framework, target_framework, source_file_path, target_file_path):
    """
    对单个文件进行迁移
    :param source_framework:
    :param target_framework:
    :param source_file_path: 原始文件路径
    :param target_file_path: 目标文件路径
    :return:
    """
    print('>>>>>>>>>>>>>>>>>>')
    print('migration: ' + source_file_path)


    # if source_framework != 'PyTorch' and target_framework != 'PyTorch':
    #     processor1 = transformers.ASTProcessor(source_file_path, '../datasets/operators_tmp/test.py')
    #     transformer1 = transformers.ASTTransformer(source_framework, 'PyTorch')
    #     # processor.print_ast()
    #     # 迁移
    #     transformer1.visit_Module(processor1.ast_root)
    #     processor1.write_target_code(processor1.ast_to_code())
    #     # processor.print_ast()
    #     # 输出目标文件
    
    #     processor2 = transformers.ASTProcessor('../datasets/operators_tmp/test.py', target_file_path)
    #     transformer2 = transformers.ASTTransformer('PyTorch', target_framework)
    #     transformer2.visit_Module(processor2.ast_root)
    #     transformer1.print_info()
    #     code = mark_unsupport(source_framework,processor2.ast_to_code())
    #     processor2.write_target_code(code)

    #     # processor2.write_target_code(processor2.ast_to_code())

    # 生成代码迁移的处理器和迁移器
    processor = transformers.ASTProcessor(source_file_path, target_file_path)
    transformer = transformers.ASTTransformer(source_framework, target_framework)
    # processor.print_ast()
        # 迁移
    transformer.visit_Module(processor.ast_root)
    # processor.print_ast()
    # # 输出目标文件
    transformer.print_info()
    code = mark_unsupport(source_framework,processor.ast_to_code())
    processor.write_target_code(code)

def transform_project(source_framework, target_framework, source_folder_name, new_folder_name=None):
    """
    对整个项目文件夹进行迁移
    :param source_framework:
    :param target_framework:
    :param source_folder_name: 原始文件夹
    :param new_folder_name:
    :return:
    """
    # 检查原始文件夹是否存在，如果不存在则打印一条消息并返回
    if not os.path.exists(source_folder_name):
        print("原始文件夹不存在")
        return

    # 如果传参没有传目标文件夹名称
    if new_folder_name is None:
        # 构建新文件夹的名称
        new_folder_name = source_folder_name + "_new"

    # 检查新文件夹是否存在，如果存在先删除
    if os.path.exists(new_folder_name):
        shutil.rmtree(new_folder_name)
    # 创建新文件夹
    os.makedirs(new_folder_name)

    # 遍历原始文件夹中的所有子文件夹和文件,os.walk自带递归功能
    for dirpath, dirnames, filenames in os.walk(source_folder_name):
        # 计算当前子文件夹相对于原始文件夹的相对路径
        relative_path = os.path.relpath(dirpath, source_folder_name)
        # 构建了新文件夹中对应的子文件夹的路径
        new_dirpath = os.path.join(new_folder_name, relative_path)
        # 检查新文件夹中对应的子文件夹是否存在，如果不存在则创建它
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)
        # 历当前文件夹中的所有文件
        for filename in filenames:
            # 构建了新文件夹中对应文件的路径
            new_file_path = os.path.join(new_dirpath, filename)
            # 创建一个空文件
            open(new_file_path, 'a').close()
            # 对当前文件进行迁移
            if filename != '__init__.py' and filename[-3:] == '.py':
                # 计算原始文件路径
                source_file_path = os.path.join(source_folder_name, relative_path, filename)
                transform_file(source_framework, target_framework, source_file_path, new_file_path)

def mark_unsupport(source_framework,code):
    if source_framework == 'PyTorch':
        prefix = 'torch'
    elif source_framework == 'MindSpore':
        prefix = 'mindspore'
    elif source_framework == 'PaddlePaddle':
        prefix = 'paddle'
    lines = code.split("\n")
    mark_next_line = False
    in_str = False
    bracket_num = 0
    for i, line in enumerate(lines):
        rm_str_line = re.sub(r"[\"]{3}[^\"]+[\"]{3}", "", line)
        rm_str_line = re.sub(r"[\"]{1}[^\"]+[\"]{1}", "", rm_str_line)
        rm_str_line = re.sub(r"[\']{1}[^\']+[\']{1}", "", rm_str_line)

        pre_in_str = in_str
        if rm_str_line.count('"""') % 2 != 0:
            in_str = not in_str
        if pre_in_str or in_str:
            continue

            # paddle.add(paddlenlp.
            #   transformers.BertTokenizer
        pre_bracket_num = bracket_num
        bracket_num += rm_str_line.count("(")
        bracket_num -= rm_str_line.count(")")
        if pre_bracket_num > 0:
            continue
        
        if re.match("^import.*",rm_str_line):
            continue

        if rm_str_line.startswith(f"{prefix}."):
            lines[i] =  line + "    # 当前算子不支持转换"
            

        # model_torch.npy

        if re.match(r".*[^\w\.]{1}%s\." % prefix, rm_str_line):
            lines[i] = line + "    # 当前算子不支持转换"

    return "\n".join(lines)

if __name__ == '__main__':
    # 其他文件夹下的相对路径 可以
    # transform_project('PyTorch', 'PaddlePaddle', "D:/Program/okgct/datasets/backend/project", None)
    # transform_project('PaddlePaddle', 'PyTorch', '../datasets/operators', None)
    # 绝对路径 可以
    # transform_project('D:\Program\ProgramPy\ProgramAST\okgct\datasets\operators')
    # transform_file('PyTorch', 'PaddlePaddle', '../test_case/torch/DenseNet.py', '../out_code/DenseNet__paddle.py')
    transform_file('PaddlePaddle', 'MindSpore', '../test_case/Paddle/vgg16.py', '../out_code/paddle2mindspore/vgg16.py')