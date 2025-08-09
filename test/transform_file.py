import sys
sys.path.append('..')
from conversion import transformers


def transform_file(argv):
    """
    对单个文件进行迁移
    sys.argv[1] : source_framework
    sys.argv[2] : target_framework
    sys.argv[3] : source_file_path 原始文件路径
    sys.argv[4] : target_file_path 目标文件路径
    sys.argv[5] : 源抽象语法树保存路径
    sys.argv[6] : 目标抽象语法树保存路径
    :return:
    """
    if sys.argv[1] != 'PyTorch' and sys.argv[2] != 'PyTorch':
        # 1
        processor1 = transformers.ASTProcessor(sys.argv[3], "D:/Program/okgct/datasets/backend/source_code_tmp.py")
        processor1.ast_to_file(sys.argv[5])

        transformer1 = transformers.ASTTransformer(sys.argv[1], 'PyTorch')
        transformer1.visit_Module(processor1.ast_root)
        processor1.write_target_code(processor1.ast_to_code())

        # 2
        processor2 = transformers.ASTProcessor("D:/Program/okgct/datasets/backend/source_code_tmp.py", sys.argv[4])

        transformer2 = transformers.ASTTransformer('PyTorch', sys.argv[2])
        transformer2.visit_Module(processor2.ast_root)

        processor2.write_target_code(processor2.ast_to_code())
        processor2.ast_to_file(sys.argv[6])
    else:
        # 生成代码迁移的处理器和迁移器
        processor = transformers.ASTProcessor(sys.argv[3], sys.argv[4])
        transformer = transformers.ASTTransformer(sys.argv[1], sys.argv[2])
        processor.ast_to_file(sys.argv[5])

        transformer.visit_Module(processor.ast_root)
        processor.write_target_code(processor.ast_to_code())
        processor.ast_to_file(sys.argv[6])


if __name__ == '__main__':
    transform_file(sys.argv)