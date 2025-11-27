#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	有向图类（无向图最短路单独设置）模块
#
###########################################################################################################################

from    .create_igraph      import  create_igraph

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   最短路 默认无向图
#
def get_shortest_paths(edges, source=None, to=None, weights=None, undirected=True) :
    g = create_igraph(edges=edges)

    if not source : source = [v.index for v in g.vs if g.degree(v, mode='in')==0]       # 全部源点
    elif type(source) == str : source = [g.vs.select(id=source)[0].index]    # 给定单一源点
    elif type(source) == list : source = [g.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
    if not to : to = [v.index for v in g.vs if g.degree(v, mode='out')==0]
    elif type(to) == str : to = [g.vs.select(id=to)[0].index]
    elif type(to) == list : to = [g.vs.select(id=vid)[0].index for vid in to]

    if not weights : weights = [1]*len(g.es)        # 与edges一致, 默认=1


    # !!!!!! 注意，无向图更改位置不能在计算源汇点之前
    if undirected : g.to_undirected(mode=False)        # 有向图变无向图

    paths = [] # 最短路径集合
    for v1 in source:
        for v2 in to:
            paths_ = g.get_shortest_paths(v1,v2,output='epath',weights=weights,algorithm="bellman_ford")
            paths_1 = [[g.es[e]['id'] for e in path] for path in paths_]
            paths += paths_1
    return paths
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


"""
def get_shortest_paths(v, to=None, weights=None, mode='out', output='vpath'):
计算从图中一个给定节点到该节点的最短路径。
参数
v           计算路径的源/目的地
to          到一个顶点选择器，描述计算路径的目标/源。这可以是单个顶点ID、顶点ID列表、单个顶点名称、顶点名称列表或一个VertexSeq对象。None表示所有的顶点。
weights     权值列表中的边权值或包含边权值的边属性的名称。如果为None，则假设所有边的权值相等。
mode        设置路径的方向性模式。“in”表示计算进来的路径，“out”表示计算出去的路径，“all”表示两者都计算。
output      输出决定了应该返回什么。如果这是“vpath”，将返回一个顶点id列表，每个目标顶点有一条路径。对于不连接的图，列表中的一些元素可能是空的。注意，在mode="in"的情况下，路径中的顶点将以相反的顺序返回。如果output="epath"，则返回边id而不是顶点id。
返回
请参阅输出参数的文档。
"""

