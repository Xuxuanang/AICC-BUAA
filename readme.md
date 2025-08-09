# OKGCT
**O**perator **K**nowledge **G**raph **C**ode **T**ransformer
本项目的项目目标是在昇腾处理器环境中基于**MindSpore AI框架**开发**模型迁移工具及AI模型**。·

## 项目文件架构
1. okg
    okg文件是算子知识图谱的构建以及与迁移脚本交互的文档。
   1. scripts文件夹中存放插入cypher语句的脚本文件，其中insert文件是将映射文件批量导入图数据库的脚本；match是连接图数据库并提供给迁移脚本调用相关图谱映射信息的文件。
   2. cypher文件夹中存放各种算子及其映射关系的cypher语句。
2. conversion
    transformers是迁移程序的主体程序，X2mindspore_mapper中存有一些提供给迁移程序使用的辅助迁移函数。
3. test
    test文件夹中存放有测试函数以及源与目标模型代码文件，其中source_code中存放源代码文件，target_code中存放目标代码文件，main文件是测试文件。

## 运行步骤
1. 运行neo4j并创建图数据库
2. 修改okg文件夹下的insert文件中的连接数据库的用户名密码及端口号
3. 运行insert文件，将算子映射信息导入图数据库
4. 在test文件夹下的运行main文件，即可在target_code文件夹中生成对应的目标模型代码