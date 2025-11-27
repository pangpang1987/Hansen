#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	有向图类模块
#
###########################################################################################################################

from    .create_igraph  import  create_igraph

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   最长路  权重为负 必须有向图, 无向图出错 无权重默认1结果是最短路
#
def get_longest_paths(edges, source=None, to=None, weights=None):

    g = create_igraph(edges=edges)


    if not source : source = [v.index for v in g.vs if g.degree(v, mode='in')==0]       # 全部源点
        
    elif type(source) == str : source = [g.vs.select(id=source)[0].index]    # 给定单一源点
        
    elif type(source) == list : source = [g.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
        
    if not to : to = [v.index for v in g.vs if g.degree(v, mode='out')==0]
    elif type(to) == str : to = [g.vs.select(id=to)[0].index]
    elif type(to) == list : to = [g.vs.select(id=vid)[0].index for vid in to]


    # 权重取负
    if weights : weights = list(map(lambda w : -1*w, weights))
    else : weights = [-1]*len(g.es)


    paths = [] # 最短路径集合
    for v1 in source:
        for v2 in to:
            paths_ = g.get_shortest_paths(v1,v2,output='epath',weights=weights,algorithm="bellman_ford")
            paths_1 = [[g.es[e]['id'] for e in path] for path in paths_]
            paths += paths_1
    return paths

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   所有点对最长路
#
#   要点 :
#   （1）必须有向图, 无向图出错
#   （2）权重内部改为负
#   （3）权重可以 <=0, 无权重默认为1, 结果是最短路
#   （4）可以并联
def get_longest_path(edges, source=None, to=None, weights=None):

    paths = get_longest_paths(edges=edges, source=source, to=to, weights=weights)

    maxPath = None
    maxH = float('-inf')        # 负无穷大

    if weights :                                    # 给定权重
        weights_ = {**{edges[i]['id']:weights[i] for i in range(len(edges))} }
        for path in paths :
            pathH = 0
            for eid in path :
                pathH += weights_[eid]
            if pathH > maxH :
                maxH = pathH
                maxPath = path
    else :                                          # 没有权重，按通路分支数计算权重
        for path in paths :
            if len(path) > maxH :
                maxH = len(path)
                maxPath = path
    return maxH, maxPath
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
