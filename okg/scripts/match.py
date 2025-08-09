from py2neo import Graph


class OKG:
    def __init__(self, username='neo4j', password='613613zzk'):
        self.username = username
        self.password = password
        self.my_graph = self.connect_neo4j(username=self.username, password=self.password)

    @staticmethod
    def connect_neo4j(username: str, password: str):
        """
        Connect to neo4j with a username and password
        通过用户名密码连接到neo4j
        :param username:
        :param password:
        :return:
        """
        _graph = Graph("bolt://neo4j@localhost:7687", auth=(username, password))
        return _graph

    @staticmethod
    def generate_property_string(property_map: dict) -> str:
        """
        Return property of cql statement
        返回格式化的属性
        :param property_map:
        :return:
        """
        if property_map == {}:
            return "{}"
        property_string = "{"
        for key, value in property_map.items():
            property_string += key + ":" + " \'" + str(value) + "\'" + ","
        property_string = property_string[:-1]
        property_string += "}"
        return property_string

    def get_operator_id(self, framework_name, operator_name: str) -> int:
        """
        Return the node id of the operator from the given framework name and operator name
        根据给定的框架名称和算子名称获取算子的节点id，如果不存在则返回-1
        :param framework_name: the given framework name
        :param operator_name: the given operator name
        :return: the operator node
        """
        cql = 'MATCH (n:operator)' \
              'WHERE n.full_name = \"{}\" ' \
              'RETURN id(n)' \
            .format(operator_name)
        node = self.my_graph.run(cql).data()
        if not node:
            return -1
        return node[0]['id(n)']
        # nodes = operator_name.split('.')  # list of node names

        # # from_labels
        # framework_node_labels = 'framework'
        # # from_properties
        # framework_node_properties = dict()
        # framework_node_properties['name'] = framework_name

        # # get the framework node id
        # cur_id = self.get_node_id(framework_node_labels, framework_node_properties)
        # for i in range(len(nodes)):
        #     if cur_id == -1:
        #         return -1
        #     # source_id
        #     source_id = cur_id
        #     # relationship_type
        #     if i == 0:  # 1 framework -> class
        #         relationship_type = 'classOfFramework'
        #     elif i == len(nodes) - 1:  # 3 class -> operator
        #         relationship_type = 'operatorOfClass'
        #     else:  # 2 class -> subClass
        #         relationship_type = 'subClassOfClass'
        #     # relationship_properties
        #     relationship_properties = dict()
        #     relationship_properties['name'] = nodes[i]
        #     cur_id = self.get_target_id_by_source_id_and_rel(source_id, relationship_type, relationship_properties)
        return cur_id

    def get_operator_full_name(self, operator_id):
        """
        根据算子id获取算子的完整名称
        :param operator_id:
        :return:
        """
        cql = 'MATCH (n) ' \
              'WHERE id(n) = {} ' \
              'RETURN n' \
            .format(operator_id)
        node = self.my_graph.run(cql).data()
        return node[0]['n'].get('full_name')

    def get_parameter_info(self, parameter_id) -> dict:
        """
        根据参数id获取参数的信息
        :param parameter_id:
        :return:
        """
        cql = 'MATCH (n) ' \
              'WHERE id(n) = {} ' \
              'RETURN n' \
            .format(parameter_id)
        node = self.my_graph.run(cql).data()
        parameter_info = {}
        for key in node[0]['n'].keys():
            parameter_info[key] = node[0]['n'][key]
        return parameter_info

    def get_parameters(self, operator_id):
        """
        Return the argument dict and list from the given operator id
        根据给定的算子id获得参数字典和列表
        :param operator_id: the framework name
        :return: parameters of the given operator
        """
        # relationship_type
        relationship_type = 'parameterOfOperator'
        # cql
        cql = 'MATCH (m)-[r: {}]-(n) ' \
              'WHERE id(m) = {} ' \
              'RETURN n' \
            .format(relationship_type, operator_id)
        nodes = self.my_graph.run(cql).data()
        #
        parameters_dict = {}
        for i in range(len(nodes)):
            node = nodes[i]['n']
            
            parameter_dict = {}
            for e in node.keys():
                parameter_dict[e] = node[e]
            
            parameters_dict[node['name']] = parameter_dict

        #删除不包含order属性的参数
        keys_to_remove = [key for key, value in parameters_dict.items() if 'order' not in value]
        for key in keys_to_remove:
            del parameters_dict[key]

        parameters_list = [0] * len(nodes)
        sorted_parameters = sorted(parameters_dict.items(), key=lambda item: item[1]['order'])
        for i in range(len(sorted_parameters)):
            parameters_list[i] = sorted_parameters[i][0]
        # for key,value in parameters_dict.items():
        #     order = int(value["order"])
        #     parameters_list[order]=key

        return parameters_dict, list(parameters_dict.keys())

    def get_inputs(self, operator_id):
        """
        根据给定的算子id获得input字典和列表
        :param operator_id:
        :return:
        """
        # relationship_type
        relationship_type = 'inputOfOperator'
        # cql
        cql = 'MATCH (m)-[r: {}]-(n) ' \
              'WHERE id(m) = {} ' \
              'RETURN n' \
            .format(relationship_type, operator_id)
        nodes = self.my_graph.run(cql).data()
        #
        inputs_dict = {}
        for i in range(len(nodes)):
            node = nodes[i]['n']
            parameter_dict = {}
            for e in node.keys():
                parameter_dict[e] = node[e]
            inputs_dict[node['input_order']] = parameter_dict
        inputs_list = sorted(inputs_dict.items(), key=lambda x: x[0])
        inputs_dict = {}
        for e in inputs_list:
            inputs_dict[e[1]['name']] = e[1]
        return inputs_dict, list(inputs_dict.keys())

    def get_target_node_info(self, source_node_id, source_node_labels, target_framework, target_version, flag=0):
        """
        Return the id of the target framework node
        based on the id and labels of the source framework node and the name of the target framework
        根据源框架节点的id、标签和目标框架名称，获得目标框架节点的相关信息：0：返回节点id；1：返回节点属性字典
        :param source_node_id:
        :param source_node_labels:
        :param target_framework:
        :param target_version:
        :param flag: 0：返回节点id；1：返回节点属性字典
        :return:
        """
        # relationship_type
        relationship_type = ''
        if source_node_labels == 'operator':
            relationship_type = 'equivalentOperator'
        elif source_node_labels == 'member_function':
            relationship_type = 'equivalentMemberFunction'
        # relationship_properties
        relationship_properties = dict()
        relationship_properties['framework_name'] = target_framework
        # relationship_properties['version'] = target_version
        relationship_properties = self.generate_property_string(relationship_properties)
        # cql
        cql = ''
        if flag == 0:
            cql = 'MATCH (m)-[r: {} {}]-(n) ' \
                  'WHERE id(m) = {} ' \
                  'RETURN id(n)' \
                .format(relationship_type, relationship_properties, source_node_id)
        elif flag == 1:
            cql = 'MATCH (m)-[r: {} {}]-(n) ' \
                  'WHERE id(m) = {} ' \
                  'RETURN n' \
                .format(relationship_type, relationship_properties, source_node_id)
        nodes = self.my_graph.run(cql).data()
        if flag == 0:
            if not nodes:
                return -1
            return nodes[0]['id(n)']
        elif flag == 1:
            if not nodes:
                return None
            else:
                res = {}
                for e in nodes[0]['n']:
                    res[e] = nodes[0]['n'][e]
                return res

    def get_member_function_id(self, framework_name, member_function_name: str) -> int:
        """
        Return the node id of a given member function by the framework name and the name of the member function node
        通过成员函数节点所处的框架名称和成员函数名称，获取给定成员函数的节点id
        :param framework_name:
        :param member_function_name:
        :return:
        """
        # from_labels
        node_labels = 'member_function'
        # from_properties
        node_properties = dict()
        node_properties['framework'] = framework_name
        node_properties['name'] = member_function_name

        # get the framework node id
        node_id = self.get_node_id(node_labels, node_properties)
        return node_id

    def get_label(self, id):
        cql = 'MATCH (n) ' \
              'WHERE id(n) = {} ' \
              'RETURN n' \
            .format(id)
        node = self.my_graph.run(cql).data()
        return str(node[0]['n'].labels)[1:]

    def get_mapping(self, id):
        """
        获取节点的“mapping”值
        :param id:  节点id
        :return:
        """
        return self.get_node_property(id, "mapping")

    def get_distinction(self, id):
        """
        获取节点的“distinction”值
        :param id:  节点id
        :return:
        """
        return self.get_node_property(id, "distinction")

    def get_name(self, id):
        """
        获取节点的“name”值
        :param id:  节点id
        :return:
        """
        return self.get_node_property(id, "name")

    def get_operator_full_name(self, id):
        """
        获取节点的“fullname”值
        :param id:  节点id
        :return:
        """
        return self.get_node_property(id, "full_name")

    def get_node_property(self, id, property_name):
        """
        基函数
        获取节点的某个属性
        :param id: 节点id
        :param property_name: 属性的名称
        :return: 属性的值
        """
        cql = 'MATCH (n) ' \
              'WHERE id(n) = {} ' \
              'RETURN n' \
            .format(id)
        node = self.my_graph.run(cql).data()
        if property_name in node[0]['n'].keys():
            return node[0]['n'].get(property_name)
        return None

    def get_custom_operator_name(self, source_name, target_framework):
        """
        获取source_name算子对应目标框架的自定义算子名称
        :param source_name:
        :param target_framework:
        :return:
        """
        source_id = self.get_node_id('operator', {'full_name': source_name})
        target_id = self.get_target_id_by_source_id_and_rel(source_id, 'customOperator',
                                                            {'framework_name': target_framework})
        return self.get_name(target_id)

    def get_parameter_id(self, operator_id, parameter_name):
        """
        获取算子的参数节点id
        :param operator_id:
        :param parameter_name:
        :return:
        """
        parameter_id = self.get_target_id_by_source_id_and_rel(
            operator_id,
            'parameterOfOperator',
            {'name': parameter_name}
        )
        return parameter_id

    def get_input_id(self, operator_id, input_name):
        """
        获取算子的参数节点id
        :param operator_id:
        :param input_name:
        :return:
        """
        input_id = self.get_target_id_by_source_id_and_rel(
            operator_id,
            'inputOfOperator',
            {'name': input_name}
        )
        return input_id

    def get_equivalent_operator_id(self, source_id, target_framework):
        """
        获取源算子对应的目标算子id
        :param source_id:
        :param target_framework:
        :return:
        """
        target_id = self.get_target_id_by_source_id_and_rel(
            source_id,
            'equivalentOperator',
            {'framework_to': target_framework}
        )
        return target_id

    def get_equivalent_parameter_id(self, source_parameter_id, target_framework):
        """
        获取源参数对应的目标参数节点id
        :param source_parameter_id:
        :param target_framework:
        :return:
        """
        target_parameter_id = self.get_target_id_by_source_id_and_rel(
            source_parameter_id,
            'equivalentParameter',
            {'framework_from': target_framework}
        )
        return target_parameter_id

    def get_target_id(self, source_parameter_id, target_framework):
        """
        获取源参数对应的目标参数节点id
        :param source_parameter_id:
        :param target_framework:
        :return:
        """
        target_id = self.get_target_id_by_source_id_and_rel(
            source_parameter_id,
            'partOfNNOperator',
            {'framework_name': target_framework}
        )
        return target_id

    def get_node_id(self, node_labels, node_properties) -> int:
        """
        基函数
        Return the id of a node through its labels and properties
        通过节点的标签和属性获得节点的id，如果不存在则返回-1
        :param node_labels:
        :param node_properties:
        :return:
        """
        node_properties = self.generate_property_string(node_properties)
        cql = 'MATCH (n: {} {}) ' \
              'RETURN id(n)' \
            .format(node_labels, node_properties)
        node = self.my_graph.run(cql).data()  # ob
        if not node:
            return -1
        return node[0]['id(n)']

    def get_equivalent_parameter_type(self, source_parameter_id, target_parameter_id, target_framework):
        """
        获取源算子的参数节点与目标算子的参数节点之间的迁移关系，如parameter2parameter、parameter2function、function2parameter
        :param source_parameter_id:
        :param target_parameter_id:
        :param target_framework:
        :return:
        """
        relationship_properties = self.generate_property_string({'framework_name': target_framework})
        cql = 'MATCH (m)-[r: equivalentParameter {}]-(n) ' \
              'WHERE id(m) = {} and id(n) = {} ' \
              'RETURN r' \
            .format(relationship_properties, source_parameter_id, target_parameter_id)
        node = self.my_graph.run(cql).data()
        if not node:
            return -1
        return node[0]['r'].get('type')

    def get_function_of_parameter(self, target_parameter_id, target_parameter_name):
        """
        获取参数节点的函数名称
        :param target_parameter_id:
        :param target_parameter_name:
        :return:
        """
        relationship_properties = self.generate_property_string({'name': target_parameter_name})
        cql = 'MATCH (m)-[r: parameterOfFunction {}]-(n) ' \
              'WHERE id(n) = {} ' \
              'RETURN m' \
            .format(relationship_properties, target_parameter_id)
        node = self.my_graph.run(cql).data()
        if not node:
            return -1
        return node[0]['m'].get('name')

    def get_target_id_by_source_id_and_rel(self, source_id, relationship_type, relationship_properties) -> int:
        """
        基函数
        通过源点的id和关系的类型及属性获得目标点的id，如果不存在则返回-1
        return the target node id from the source node id and the relationship
        :param source_id:
        :param relationship_type:
        :param relationship_properties:
        :return:
        """
        relationship_properties = self.generate_property_string(relationship_properties)
        cql = 'MATCH (m)-[r: {} {}]-(n) ' \
              'WHERE id(m) = {} ' \
              'RETURN id(n)' \
            .format(relationship_type, relationship_properties, source_id)
        node = self.my_graph.run(cql).data()
        if not node:
            return -1

        return node[0]['id(n)']


if __name__ == '__main__':
    graph = OKG(username='neo4j', password='613613zzk')
    info = graph.get_function_of_parameter(43, 'operations')
    print(info)
