#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph模块
#
###########################################################################################################################

import  igraph  as  ig

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建igraph
#
def create_igraph(edges, directed=True) -> tuple:

    # 1. 创建有向图igraph
    dGraph = ig.Graph(directed=directed)

    # 2. 实例化iGraph

    # （1）提取节点设置节点索引映射
    ss = list(map(lambda e: e["s"], edges))
    ts = list(map(lambda e: e["t"], edges))
    vs = list(set(ss + ts))

    # （2）添加节点, 设置节点索引映射
    dGraph.add_vertices(len(vs))  # 添加 5 个节点
    dGraph.vs["id"] = vs  # 为节点设置id属性
    vid_to_index = {**{v["id"] : v.index  for v in dGraph.vs}}
    index_to_vid = {**{v.index : v['id'] for v in dGraph.vs}}

    # （3）添加边
    dGraph.add_edges([(vid_to_index[e["s"]], vid_to_index[e["t"]]) for e in edges])  # 添加边
    dGraph.es["id"] = [e["id"] for e in edges]  # 为边设置权重属性
    eid_to_index = {**{e["id"] : e.index  for e in dGraph.es}}
    index_to_eid = {**{e.index : e['id'] for e in dGraph.es}}

    return dGraph, vid_to_index, index_to_vid, eid_to_index, index_to_eid
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
