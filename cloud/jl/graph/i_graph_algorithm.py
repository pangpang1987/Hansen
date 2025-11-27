#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	通风网络图模块
#
###########################################################################################################################

from    ._i_graph.create_igraph      import  create_igraph           as  _create_igraph_

from    contextlib                      import  contextmanager


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   图类
#
class IGraph :


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   无向图最短路
    #   备注：无向图权重 weght>=0, 可以使用正无穷
    def get_shortest_paths_undirected(
            self, 
            edges,
            source=None,    # str,list, None, None：全风网源点
            to=None,    # str,list, None, None：全风网汇点
            weights=None    # list, 与edges序列一致, Node,分支权重默认1,权重可以为负值
            # undirected=True # 默认无向图
        ) :
    
        jG = _create_igraph_(edges)

        if not source : source = [v.index for v in jG.vs if jG.degree(v, mode='in')==0]       # 全部源点
        elif type(source) == str : source = [jG.vs.select(id=source)[0].index]    # 给定单一源点
        elif type(source) == list : source = [jG.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
        if not to : to = [v.index for v in jG.vs if self.dg.degree(v, mode='out')==0]
        elif type(to) == str : to = [jG.vs.select(id=to)[0].index]
        elif type(to) == list : to = [jG.vs.select(id=vid)[0].index for vid in to]

        if not weights : weights = [1]*len(jG.es)        # 与edges一致, 默认=1


        # !!!!!! 注意，无向图更改位置不能在计算源汇点之前
        # if undirected : self.dg.to_undirected(mode=False)        # 有向图变无向图
        jG.to_undirected(mode=False)        # 有向图变无向图

        paths = [] # 最短路径集合
        for v1 in source:
            for v2 in to:
                paths_ = jG.get_shortest_paths(v1,v2,output='epath',weights=weights,algorithm="bellman_ford")
                paths_1 = [[jG.es[e]['id'] for e in path] for path in paths_]
                paths += paths_1
        return paths

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   有向图最短路
    #   备注：无向图权重 weght>=0, 可以使用正无穷
    def get_shortest_paths(
            self, 
            source  =   None,       # str,list, None, None：全风网源点
            to      =   None,       # str,list, None, None：全风网汇点
            weights =   None        # list, 与edges序列一致, Node,分支权重默认1,权重可以为负值
            # undirected=True # 默认无向图
        ) :

        if not source : source = [v.index for v in self.dg.vs if self.dg.degree(v, mode='in')==0]       # 全部源点
        elif type(source) == str : source = [self.dg.vs.select(id=source)[0].index]    # 给定单一源点
        elif type(source) == list : source = [self.dg.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
        if not to : to = [v.index for v in self.dg.vs if self.dg.degree(v, mode='out')==0]
        elif type(to) == str : to = [self.dg.vs.select(id=to)[0].index]
        elif type(to) == list : to = [self.dg.vs.select(id=vid)[0].index for vid in to]

        if not weights : weights = [1]*len(self.dg.es)        # 与edges一致, 默认=1

        paths = [] # 最短路径集合
        for v1 in source:
            for v2 in to:
                paths_ = self.dg.get_shortest_paths(v1,v2,output='epath',weights=weights,algorithm="bellman_ford")
                paths_1 = [[self.dg.es[e]['id'] for e in path] for path in paths_]
                paths += paths_1
        return paths

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最长路远离：将权重设置为负，必须是有向图, 无向图出错，无权重默认1结果是最短路
    #   要点 :
    #   （1）必须有向图, 无向图出错
    #   （2）权重内部改为负
    #   （3）权重可以 <=0, 无权重默认为1, 结果是最短路
    #   （4）可以并联
    def get_longest_paths(self, source=None, to=None, weights=None):

        if not source : source = [v.index for v in self.dg.vs if self.dg.degree(v, mode='in')==0]       # 全部源点
        
        elif type(source) == str : source = [self.dg.vs.select(id=source)[0].index]    # 给定单一源点
        
        elif type(source) == list : source = [self.dg.vs.select(id=vid)[0].index for vid in source]   # 给定源点列表（可以是单一源点）
        
        
        if not to : to = [v.index for v in self.dg.vs if self.dg.degree(v, mode='out')==0]
        elif type(to) == str : to = [self.dg.vs.select(id=to)[0].index]
        elif type(to) == list : to = [self.dg.vs.select(id=vid)[0].index for vid in to]


        # 权重取负
        if weights : weights = list(map(lambda w : -1*w, weights))
        else : weights = [-1]*len(self.dg.es)


        paths = [] # 最短路径集合
        for v1 in source:
            for v2 in to:
                paths_ = self.dg.get_shortest_paths(v1,v2,output='epath',weights=weights,algorithm="bellman_ford")
                paths_1 = [[self.dg.es[e]['id'] for e in path] for path in paths_]
                paths += paths_1
        return paths

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
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
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
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

 
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
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
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


            
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
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
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   igraph法找全部通路，可以是无向图，如果有向图没有通路或内存超限返回空列表（内部有异常中断处理） []
    def get_paths_igraph111(self, edges, source=None, to=None, undirected=False) -> list :
        g = G()
        return g.get_paths(edges=edges, source=source,to=to,undirected=undirected,cutoff=-1, mode='out')
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索全部通路    主要用于确定系统复杂度
    def get_all_paths_depth_first_search(self, source=None, to=None, mode='number') -> list :
        # path通路，number复杂度，all全部
        return _get_all_paths_depth_first_search_(self.edges, starts=source, targets=to, mode='number')




    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   截流（用于密闭、盲巷处置）                  暂未使用JianGraph的深度优先搜索
    def closure(self, edges, giveways=[]) ->list : return _closure_(edges=edges, giveways=giveways)


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   JianGraph上下文
#
@contextmanager
def context_i_graph_algorithm(edges):
    iGraph = IGraph(edges)
    yield iGraph
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€









