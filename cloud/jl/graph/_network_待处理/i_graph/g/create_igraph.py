#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph模块
#
###########################################################################################################################

import  igraph              as      ig
from    .graph_algrithom    import  *


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建igraph函数
#
def create_igraph(edges):
    
    # 1. 创建igraph
    g = ig.Graph(directed=True)

    # 2. 节点id（key）与igraph索引（value）映射
    all_v = all_vertexes(edges)
    mapped_nodes = list_to_index_map(all_v)

    # 3. 分支映射
    mapped_edges_nodes = edges_map(edges, mapped_nodes)

    # 4. 所有节点属性（id）
    vertices = list(mapped_nodes.keys())

    # 5. 创建节点索引
    node_num = len(mapped_nodes)

    # 6. 添加节点
    g.add_vertices(node_num)

    # 7. 节点id
    g.vs["id"] = vertices

    # 8. 添加边
    g.add_edges(mapped_edges_nodes['mapped'])

    # 9. 添加边eid
    g.es["id"] = mapped_edges_nodes["eids"]

    # 10. 返回igraph类对象
    return g
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

