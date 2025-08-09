import os
from py2neo import Graph


class Neo4jExample:

    def __init__(self, username='neo4j', password='613613zzk'):
        self.username = username
        self.password = password
        self.my_graph = self.connect_neo4j(username=self.username, password=self.password)

    @staticmethod
    def connect_neo4j(username: str, password: str):
        # 通过用户名密码连接到neo4j
        my_graph = Graph(
            "bolt://neo4j@localhost:7687",
            auth=(username, password)
        )
        return my_graph

    def insert_operator(self, filename):
        """

        :param filename:
        :return:
        """
        with open(filename, encoding='utf-8') as file:
            tmp_cqls = ''
            lines = file.readlines()
            for line in lines:
                if line[0] == "/":
                    continue
                line = line.rstrip("\n")
                tmp_cqls = tmp_cqls + line + ' '
            tmp_cqls = tmp_cqls.rstrip()
            cqls = tmp_cqls.split(";")
            cqls = cqls[:-1]
            for cql in cqls:
                self.my_graph.run(cql)

    def insert_operators(self, folder):
        """

        :param folder:
        :return:
        """
        files = os.listdir(folder)
        for file in files:
            if file[-3:] == "txt":
                filename = folder + '/' + file
                self.insert_operator(filename)


if __name__ == '__main__':
    graph = Neo4jExample(username='neo4j', password='613613zzk')
    # graph.insert_operators("../cypher/pytorch_paddle/module/torch.nn")
    # graph.insert_operators("../cypher/pytorch_paddle/new")
