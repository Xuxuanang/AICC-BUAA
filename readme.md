# AICC
**AI** **C**ode **C**onverter based knowledge graph
本项目是由北航团队开发的**AI模型迁移工具**。·

## 项目文件架构
1. conversion
    transformers是迁移程序的主体程序。
2. okg
   该文件夹下为从知识图谱中读取数据及算子相关信息的脚本/函数。
3. test
   该文件夹中main文件为程序主入口。
4. test_case文件夹下为可以测试的模型。

## 运行步骤
1. 运行neo4j并创建图数据库
2. 修改okg文件夹下的match文件中的连接数据库的用户名密码及端口号
3. 修改test文件夹下的main文件中输入输出路径，并运行，即可完成代码的转换。