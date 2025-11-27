#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#	JGraph	最小树模块
#
######################################################################################

#   说明:
#   （1）可以是非连通图
#   （2）可以含有单向回路
#   （3）权重须 >= 0

from    jl.jGraph.getDGraph            import  GetDGraph

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求最小树
#

def GetMinTree(topList,weights=None) :
    """计算一个图的最小生成树
    参数:
    :topList--拓扑关系列表 [{"id":"e1","s":"v1","t":"v2"}] 
    weights----一个包含图中每条边的权值的向量。None表示图是未加权的。
    return_tree ----是返回最小生成树(当return_tree为True时)还是返回最小生成树的边id
    (当return_tree为False时)。由于历史原因，默认为True，因为这个参数是在igraph 0.6中引入的。
    返回:
    如果return_tree为真，则生成树作为图形对象;如果return_tree为假，则生成树的边缘id为图形对象。
    """
    # 1. 创建有向图
    dg,map_edge,map_node = GetDGraph(topList)

    if weights :
        weights = [weights[e["id"]] for e in dg.es]
    tree = dg.spanning_tree(weights=weights,return_tree=False)

    return [dg.es[i]["id"] for i in tree]
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


