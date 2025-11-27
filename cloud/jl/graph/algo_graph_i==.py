#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph 模块
#
###########################################################################################################################

from .i.create_igraph               import  create_igraph
from .i.topology_check              import  topology_check  as  _topology_check_
from .i.get_unidirectional_loops    import  get_unidirectional_loops    as  _get_unidirectional_loops_
from .i.get_spanning_tree           import  get_spanning_tree   as  _get_spanning_tree_


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   有向图类
#
class GraphI :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   初始化函数
    def __init__(self, edges):
        self.edges = edges
        self.dGraph, self.mapv, self.mape = create_igraph(self.edges)

    
   

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@








#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取节点入边
def get_node_in_edge(self) -> dict :
        return {**{v['id'] : dGraph.es[dGraph.incident(v,mode='in')]['id'] for v in dGraph.vs}}

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取节点出边
def get_node_out_edge(self) -> dict :
        return {**{v['id'] : dGraph.es[dGraph.incident(v,mode='out')]['id'] for v in dGraph.vs}}


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   有向图最短路
#   备注：无向图权重 weght>=0, 可以使用正无穷
#   场景：找回路、避灾路线
def get_shortest_paths(
    dGraph, 
    source  =   None,       # str,list, None, None：全风网源点
    to      =   None,       # str,list, None, None：全风网汇点
    weights =   None,        # list, 与edges序列一致, Node,分支权重默认1,权重可以为负值
    mode    =   "ALL"       # 遍历方向，适用于有向图. "ALL"：忽略边的方向。"OUT"：沿着边的方向（默认）。"IN"：逆着边的方向。
) :

    # 1. 源点id转索引
    # 1.1 无源点（全部源点）
    if not source : source = [v.index for v in dGraph.vs if dGraph.degree(v, mode='in')==0]       # 全部源点
    # 1.2 给定单一源点
    elif type(source) == str : source = [dGraph.vs.select(id=source)[0].index]    # 给定单一源点
    # 1.3 给定源点列表
    elif type(source) == list : source = [dGraph.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
    
    # 2. 目标点id转索引
    # 2.1 无目标点（全部目标点）
    if not to : to = to=None
    # 2.2 给定单一目标点
    elif type(to) == str : to = [dGraph.vs.select(id=to)[0].index]
    # 2.3 给定目标点列表
    elif type(to) == list : to = [dGraph.vs.select(id=vid)[0].index for vid in to]

    # if not weights : weights = [1]*len(dGraph.es)        # 与edges一致, 默认=1

    paths = [] # 最短路径集合
    for v1 in source:
        paths_ = dGraph.get_shortest_paths(v1,to=to,output='epath',weights=weights,algorithm="bellman_ford",mode=mode)
        paths_1 = [[dGraph.es[e]['id'] for e in path] for path in paths_]
        paths += paths_1
    return paths
#   get_shortest_paths() 是 igraph 库中用于计算图中最短路径的核心函数。它可以找出从一个或多个源节点到目标节点的最短路径，支持加权和无权图。
#   函数基本用法
#   Graph.get_shortest_paths(v, to=None, weights=None, mode="ALL", output="vpath", ...)
#   参数说明
#   v：源节点（单个节点或节点列表）。
#   to：目标节点（单个节点或节点列表）。默认为所有节点。
#   weights：可选参数，指定边的权重列表。若不提供，则所有边权重视为 1。
#   mode：遍历方向，适用于有向图：
#   "ALL"：忽略边的方向。
#   "OUT"：沿着边的方向（默认）。
#   "IN"：逆着边的方向。
#   output：返回路径的类型：
#   "vpath"：节点 ID 列表（默认）。
#   "epath"：边 ID 列表。
#   "both"：同时返回节点和边 ID。
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€



#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最长路远离：将权重设置为负，必须是有向图, 无向图出错，无权重默认1结果是最短路
    #   要点 :
    #   （1）必须有向图, 无向图出错
    #   （2）权重内部改为负
    #   （3）权重可以 <=0, 无权重默认为1, 结果是最短路
    #   （4）可以并联
def get_longest_paths(dGraph, source=None, to=None, weights=None):

        if not source : source = [v.index for v in dGraph.vs if dGraph.degree(v, mode='in')==0]       # 全部源点
        
        elif type(source) == str : source = [dGraph.vs.select(id=source)[0].index]    # 给定单一源点
        
        elif type(source) == list : source = [dGraph.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
        
        if not to : to = [v.index for v in dGraph.vs if dGraph.degree(v, mode='out')==0]
        elif type(to) == str : to = [dGraph.vs.select(id=to)[0].index]
        elif type(to) == list : to = [dGraph.vs.select(id=vid)[0].index for vid in to]


        # 权重取负
        if weights : weights = list(map(lambda w : -1*w, weights))
        else : weights = [-1]*len(dGraph.es)


        paths = [] # 最短路径集合
        for v1 in source:
            for v2 in to:
                paths_ = dGraph.get_shortest_paths(v1,v2,output='epath',weights=weights,algorithm="bellman_ford")
                paths_1 = [[dGraph.es[e]['id'] for e in path] for path in paths_]
                paths += paths_1
        return paths

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    #   所有点对最长路  igraph
    #
    #   要点 :
    #   （1）必须有向图, 无向图出错
    #   （2）权重内部改为负
    #   （3）权重可以 <=0, 无权重默认为1, 结果是最短路
    #   （4）可以并联
def get_longest_path(self, source=None, to=None, weights=None):

        paths = self.get_longest_paths(source=source, to=to, weights=weights)

        maxPath = None
        maxH = float('-inf')        # 负无穷大

        if weights :                                    # 给定权重
            weights_ = {**{self.edges[i]['id']:weights[i] for i in range(len(self.edges))} }
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

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先遍历搜索通路数、最大通路值、最大通路 用于校验igraph法
    #   有向图，分支权重>0，可以有单向回路，
def get_sumpath_maxh_maxpath_depth_first_traversal_search(
        self,
        source      =   None,       # None/str/list，搜索起始点
        to          =   None,       # None/str/list，搜索终点
        giveways    =   [],         # list，避让id
        weights     =   None        # None/list，与edges一致
    ) :
        if weights :edges = [dict({**e, **{'weight':weights[i]}}) for i,e in enumerate(self.edges)]
        else : edges = [dict({**e, **{'weight':1}}) for i,e in enumerate(self.edges)]
        return  _get_sumpath_maxh_maxpath_depth_first_traversal_search_(
                    edges,
                    source=source,
                    to=to,
                    giveways=giveways
                )

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    #   同向串联分支    放在jGraph
    #
def get_series_edge(self) :

        seriesv = [v.index for v in dGraph.vs if dGraph.degree(v.index,mode='in')==1 \
                   and dGraph.degree(v.index,mode='out')==1]                         # [index, ...], 串联节点索引, 入度=出度=1
        vcolor = []                                                                 # [index, ...], 着色节点
        seriess = []                                                                # [[eid1,eid2, ...], ...], 串联分支id
        def _find_in_(v0,series) :                                                  # 递归寻找入边
            if v0 in seriesv :
                ei = dGraph.incident(v0, mode='in')
                vcolor.append(v0)
                series.insert(0,ei[0])
                vi = dGraph.es[ei[0]].source
                _find_in_(vi,series)
        def _find_out_(v0,series) :
            if v0 in seriesv :
                eo = dGraph.incident(v0, mode='out')
                vcolor.append(v0)
                series.append(eo[0])
                vo = dGraph.es[eo[0]].target
                _find_out_(vo,series)

        while len(seriesv) :
            v0 = seriesv.pop(0)
            if v0 in vcolor : continue
            series = []                                         # 串联节点串联分支集合
            vcolor.append(v0)
            ei = dGraph.incident(v0, mode='in')
            eo = dGraph.incident(v0, mode='out')
            series.insert(0,ei[0])
            series.append(eo[-1])
            vi = dGraph.es[ei[0]].source
            vo = dGraph.es[eo[0]].target
            vcolor.extend([vi,vo])
            _find_in_(vi,series)
            _find_out_(vo,series)
            series_ = [dGraph.es[i]['id'] for i in series]           # index转id
            seriess.append(series_)

        return seriess
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    #   无向串联分支    用于删除掘进巷道    与get_series_edge不同在于只要单链接度=2即可
    #
def get_series_edge_undirected(edges, eids) :
        # eids 搜索基点id

        g = _create_igraph_(edges=edges)

        seriess = []

        # 找与v0节点连接，但不是e的分支
        def _find_(v0, eindex, series) :                                                  # 递归寻找入边
            # v0 - 当前节点
            # e - 当前分支
            if g.degree(v0,mode='all')==2 :
                ei = g.incident(v0, mode='in')[0]          # 分支类对象
                eo = g.incident(v0, mode='out')[0]
                if ei != eindex : e=g.es[ei]
                if eo != eindex : e=g.es[eo]
                series.append(e['id'])
                if e.source != v0 :
                    _find_(e.source, e.index, series)
                if e.target != v0 :
                    _find_(e.source, e.index, series)
            else :
                return

        for eid in eids :
            series = [eid]
            eindex = g.es.select(id=eid)[0].index
            v1index = g.es[eindex].source
            v2index = g.es[eindex].target

            _find_(v1index, eindex, series)
            _find_(v2index, eindex, series)

            seriess.append(series)
    
        # print(seriess)
        return seriess
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


            
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   汇点子网-公共分支
def get_sink_node_subnet_common_edge111(self, edges) :

        sinkNodes = self.get_topology(edges)['sinkNodes']       # list 汇点id
        j = JianGraph(edges)
        esinks = j.set_edge_associated_sink_node()              # 设置分支关联汇点

        me = {**{e['id']:e for e in edges}}

        sinkNodeSubnet = dict()
        for sinkNode in sinkNodes :
            sinkNodeSubnet[sinkNode] = list()
        commonEdges = list()
        for eid, sinks in esinks.items() :
            if len(sinks) > 1 : commonEdges.append(eid)     # 公共分支
            for sink in sinks :
                sinkNodeSubnet[sink].append(me[eid])            # 将eid加入汇点sink
        
        return sinkNodeSubnet, commonEdges
    
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   igraph法找全部通路，可以是无向图，如果有向图没有通路或内存超限返回空列表（内部有异常中断处理） []
def get_paths_igraph111(self, edges, source=None, to=None, undirected=False) -> list :
        g = G()
        return g.get_paths(edges=edges, source=source,to=to,undirected=undirected,cutoff=-1, mode='out')
    
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索全部通路    主要用于确定系统复杂度
def get_all_paths_depth_first_search(self, source=None, to=None, mode='number') -> list :
        # path通路，number复杂度，all全部
        return _get_all_paths_depth_first_search_(self.edges, starts=source, targets=to, mode='number')




#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   截流（用于密闭、盲巷处置）                  暂未使用JianGraph的深度优先搜索
def closure(self, edges, giveways=[]) ->list : return _closure_(edges=edges, giveways=giveways)


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   JianGraph上下文
#
# @contextmanager
# def context_i_graph_algorithm(edges):
#     iGraph = IGraph(edges)
#     yield iGraph
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€









