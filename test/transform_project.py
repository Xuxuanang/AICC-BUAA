import sys
sys.path.append('..')
from conversion import transformers

import os
import shutil

def transform_file(source_framework, target_framework, source_file_path, target_file_path):
    """
    对单个文件进行迁移
    :param source_framework:
    :param target_framework:
    :param source_file_path: 原始文件路径
    :param target_file_path: 目标文件路径
    :return:
    """
    if source_framework != 'PyTorch' and target_framework != 'PyTorch':
        processor1 = transformers.ASTProcessor(source_file_path, "D:/Program/okgct/datasets/backend/source_code_tmp.py")
        transformer1 = transformers.ASTTransformer(source_framework, 'PyTorch')
        # processor.print_ast()
        # 迁移
        transformer1.visit_Module(processor1.ast_root)
        processor1.write_target_code(processor1.ast_to_code())
        # processor.print_ast()
        # 输出目标文件

        processor2 = transformers.ASTProcessor("D:/Program/okgct/datasets/backend/source_code_tmp.py", target_file_path)
        transformer2 = transformers.ASTTransformer('PyTorch', target_framework)
        transformer2.visit_Module(processor2.ast_root)
        processor2.write_target_code(processor2.ast_to_code())
    else:
        # 生成代码迁移的处理器和迁移器
        processor = transformers.ASTProcessor(source_file_path, target_file_path)
        transformer = transformers.ASTTransformer(source_framework, target_framework)
        # processor.print_ast()
        # 迁移
        transformer.visit_Module(processor.ast_root)
        # processor.print_ast()
        # 输出目标文件
        processor.write_target_code(processor.ast_to_code())

def transform_project(argv):
    innner_transform_project(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


def innner_transform_project(source_framework, target_framework, source_folder_name, new_folder_name=None):
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


if __name__ == '__main__':
    transform_project(sys.argv)