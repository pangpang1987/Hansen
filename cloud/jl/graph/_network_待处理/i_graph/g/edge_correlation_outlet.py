#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph模块
#
###########################################################################################################################

from  .create_igraph      import  create_igraph
from    .graph_algrithom    import  *

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建igraph函数
#
# 关联分支对应的风井节点
# def edge_correlation_outlet(edges_data, inlets=[], outlets=[]):
def edge_correlation_outlet(net, inlets=[], outlets=[]):

    # 生成igraph图
    # g = create_igraph(edges_data)
    g=net.g
    inlets = net.get_nodes_source()
    outlets = net.get_nodes_sink()

    # 节点索引转换
    inlets_index = [g.vs.select(id=inlet)[0].index for inlet in inlets]
    outlets_index = [g.vs.select(id=outlet)[0].index for outlet in outlets]
    # 寻找与汇风节点关联的不重复的边
    associated_edges = {}
    for outlet_index in outlets_index:
        # 汇风节点
        outlet_vertx = g.vs[outlet_index]["id"]
        paths = []
        edges = set()
        for inlet_index in inlets_index:
            paths += g.get_all_simple_paths(inlet_index, outlet_index)
        for path in paths:  # 循环找到的所有关联分支
            for j in range(len(path) - 1):
                s = path[j]  # 分支始节点
                t = path[j + 1]  # 分支末节点
                # 查找节点间的边
                edge = list(set(g.es[g.incident(s)]['id']) & set(g.es[g.incident(t, mode='in')]['id']))
                for i in edge:
                    edges.add(i)  # 处理并联分支
        associated_edges[outlet_vertx]=list(edges)
    # list_of_dicts = [{'key': key, 'value': value} for key, value in original_dict.items()]
    # 边关联的回风井
    edges_to_v = {item: [key for key, value in associated_edges.items() if item in value]
            for sublist in associated_edges.values() for item in sublist}

    return [{key: value} for key, value in edges_to_v.items()]
