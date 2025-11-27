#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph模块
#
###########################################################################################################################

from    .create_igraph      import  create_igraph
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求点对通路函数（可以是无向图）      注意有向图无通路、内存超限意外终止
#
def get_paths(edges, source=None, to=None, cutoff=-1, mode='out', undirected=False) :
    g = create_igraph(edges=edges)
    if not source : source = [v.index for v in g.vs if g.degree(v, mode='in')==0]       # 全部源点
    elif type(source) == str : source = [g.vs.select(id=source)[0].index]    # 给定单一源点
    elif type(source) == list : source = [g.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
    if not to : to = [v.index for v in g.vs if g.degree(v, mode='out')==0]
    elif type(to) == str : to = [g.vs.select(id=to)[0].index]
    elif type(to) == list : to = [g.vs.select(id=vid)[0].index for vid in to]

    if undirected : g.to_undirected(mode=False)        # 有向图变无向图

    # 寻找与汇风节点关联的不重复的边
    paths_ = list()
    for v1 in source :
        # 汇风节点
        # outlet_vertx = g.vs[outlet_index]["id"]
        paths = []
        edges = set()
        for v2 in to:
            paths += g.get_all_simple_paths(v=v1, to=v2,cutoff=cutoff, mode=mode)
        # print('paths===========================',paths)

        for path in paths:  # 循环找到的所有关联分支
            path_ = list()
            for j in range(len(path) - 1):
                s = path[j]  # 分支始节点
                t = path[j + 1]  # 分支末节点
                # 查找节点间的边
                edge = list(set(g.es[g.incident(s)]['id']) & set(g.es[g.incident(t, mode='in')]['id']))
                path_.append(edge[0])               # 并联分支取首个
                for i in edge:
                    edges.add(i)  # 处理并联分支
            paths_.append(path_)

    return paths_
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

# def get_all_simple_paths(self, v, to=None, cutoff=-1, mode='out'): ¶
# Calculates all the simple paths from a given node to some other nodes (or all of them) in a graph.
# A path is simple if its vertices are unique, i.e. no vertex is visited more than once.
# Note that potentially there are exponentially many paths between two vertices of a graph, 
#   especially if your graph is lattice-like. In this case, you may run out of memory when using this function.
# Parameters
# v	the source for the calculated paths
# to	a vertex selector describing the destination for the calculated paths. This can be a single vertex ID, 
#   a list of vertex IDs, a single vertex name, a list of vertex names or a VertexSeq object. None means all the vertices.
# cutoff	maximum length of path that is considered. If negative, paths of all lengths are considered.
# mode	the directionality of the paths. "in" means to calculate incoming paths, "out" means to calculate outgoing paths, 
#   "all" means to calculate both ones.
# Returns
# all of the simple paths from the given node to every other reachable node in the graph in a list. 
#   Note that in case of mode="in", the vertices in a path are returned in reversed order!

# 计算图中从一个给定节点到其他一些节点(或所有节点)的所有简单路径。
# 如果一个路径的顶点都是唯一的，那么这个路径就是简单的，即没有一个顶点被访问超过一次。
# 请注意，图的两个顶点之间可能存在指数级的路径，特别是当你的图是类似格结构的时候。在这种情况下，您可能会在使用此函数时耗尽内存。
# 参数
# V         计算路径的来源
# to        到描述计算路径的目的地的顶点选择器。这可以是单个顶点ID、顶点ID列表、单个顶点名称、顶点名称列表或VertexSeq对象。None表示所有的顶点。
# cutoff    截断所考虑的路径的最大长度。如果是负数，则考虑所有长度的路径。
# mode      设置路径的方向性。“in”表示计算传入路径，“out”表示计算传出路径，“all”表示同时计算两者。
# Returns   图中从给定节点到其他所有可达节点的所有简单路径都放在一个列表中。注意，在mode="in"的情况下，路径中的顶点会以相反的顺序返回!


