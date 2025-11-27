#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	矿井通风网络图类    不使用节点分支类
#
###########################################################################################################################

from    .i_graph.graph              import  G
from    .j_graph.jian_graph         import  JianGraph

from    .get_topology                   import  get_topology    as  _get_topology_

from    .net.circuit                import  set_mintree_cotree_circuits   as  _set_mintree_cotree_circuits_

from    .net.closure                import  closure     as  _closure_

from    .virtual_edge               import  get_virtual_edges               as  _get_virtual_edges_

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   网络类
#
class NetworkX() :
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self) : pass

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   拓扑关系
    def get_topology(self, edges) -> dict :
        return  _get_topology_(edges)
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   判断连通图
    def is_connected(self, edges) -> bool :
        g = G()
        return g.is_connected(edges=edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取连通图
    def get_connected_graph(self,edges) -> list :
        g = G()
        return g.get_connected_graph(edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   检查是否为有向无环图directed acyclic graph, DAG不含单向回路及单节点环
    def is_dag(self, edges) -> bool :
        g = G()
        return g.is_dag(edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取单节点环
    def get_loops(self, edges) -> list :
        g = G()
        return g.get_loops(edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取节点入边
    def get_node_in_edge(self, edges) -> dict :
        g = G()
        return g.get_node_in_edge(edges=edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取节点出边
    def get_node_out_edge(self, edges) -> dict :
        g = G()
        return g.get_node_out_edge(edges=edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   闭合网络虚拟分支
    def get_virtual_edges(self, edges) -> tuple :
        return _get_virtual_edges_(edges=edges)
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最小树-余树-回路函数
    def get_mintree_cotree_circuits(self, edges, weights=None) -> tuple :
        return _set_mintree_cotree_circuits_(edges,weights)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   同向串联
    def get_series_edge(self, edges) :
        g = G()
        return g.get_series_edge(edges)
    
            
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   汇点子网-公共分支
    def get_sink_node_subnet_common_edge(self, edges) :

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
    def get_paths_igraph(self, edges, source=None, to=None, undirected=False) -> list :
        g = G()
        return g.get_paths(edges=edges, source=source,to=to,undirected=undirected,cutoff=-1, mode='out')

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最短路    默认无向图, 权重可以负值
    def get_shortest_paths(
        self, 
        edges,                      # list, [{'id':id,'s':s,'t':t},...]
        source      =   None,       # str,list, None, None：全风网源点
        to          =   None,       # str,list, None, None：全风网汇点
        weights     =   None,       # list, 与edges序列一致, Node,分支权重默认1,权重可以为负值
        undirected  =   True        # 默认无向图
    ) -> list :                     # list, [[eid1,eid2,...],...]
        g = G()
        return g.get_shortest_paths(edges=edges, source=source, to=to, weights=weights, undirected=undirected)
   
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最长路（每个点对都有一条最长路）
    #   要点 :
    #   （1）必须有向图, 无向图出错
    #   （2）权重内部改为负
    #   （3）权重可以 <=0, 无权重默认为1, 结果是最短路
    #   （4）可以并联
    def get_max_paths(self, edges, source=None, to=None, weights=None) ->list :
        g = G()
        return g.get_longest_paths(edges=edges, source=source,to=to, weights=weights)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最长路（所有点对选一条）    最长路  权重内部改为负 必须有向图, 无向图出错 无权重默认1结果是最短路
    def get_max_path(
        self, 
        edges, 
        source=None, 
        to=None, 
        weights=None        # list, [w1,w2,...]次序与edges一致，权重可以为负值
    ) -> tuple :            # (float,[eid1,eid2,...]) 
        g = G()
        return g.get_longest_path(edges=edges,source=source,to=to,weights=weights)
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   截流（用于密闭、盲巷处置）                  暂未使用JianGraph的深度优先搜索
    def closure(self, edges, giveways=[]) ->list : return _closure_(edges=edges, giveways=giveways)


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



            


    
        









#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#         # 5. 闭合处理虚拟分支
#         self.closed = closed
#         if self.closed :
#         	# 5.1 基点-源点虚拟分支
#             i = 1
#             self.es_virtual_source = list()
#             for vid in self.sourceNodes :               # 基点-源点循环
#                 #创建虚拟源支
#                 edge = {"id":'virtual_source_%d' % i,"s":"virtual","t":vid}
#                 self.es_virtual_source.append(edge)     
#                 i += 1
#             # 5.2 汇点-基点虚拟分支
#             i = 1
#             self.es_virtual_sink = list()            # 汇点-基点虚拟分支
#             for vid in self.sinkNodes :
#                 edge = {"id":'virtual_sink_%d' % i,"s":vid,"t":"virtual"}
#                 self.es_virtual_sink.append(edge)
#                 i += 1
#             # 5.3 全部虚拟分支
#             self.es_virtual_all = list()
#             self.es_virtual_all.extend(self.es_virtual_source)
#             self.es_virtual_all.extend(self.es_virtual_sink)

#         else :
#             self.es_virtual_all     =   []
#             self.es_virtual_source  =   []
#             self.es_virtual_sink    =   []

#         # 6. igraph
#         self.dg, self.edgeIdIndexMap, self.nodeIdIndexMap = _GetDGraph(self.es)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   搜索串联风路
#     def SearchSeriesEdges(self, eid=None,vid=None,graph="UNDIRECTED") :
#         seriess = list()
#         colorNodes = list()
#         colorEdges = list()

#         def _find_edge(series,eid) :
#             colorEdges.append(eid)
#             edge = self.esDict[eid]
#             s,t = edge["s"], edge["t"]
#             if s not in colorNodes and self.dgree[s]==2 :
#                 colorNodes.append(s)
#                 e0 = self.inOutEdges[s][0]
#                 e1 = self.inOutEdges[s][1]
#                 if e0 not in colorEdges :
#                     # colorEdges.append(e0)
#                     series.insert(0,e0)
#                     series = _find_edge(series,e0)
#                 if e1 not in colorEdges :
#                     # colorEdges.append(e1)
#                     series = series.insert(0,e1)
#                     _find_edge(series,e1)
#             if t not in colorNodes and self.dgree[t]==2 :
#                 colorNodes.append(t)
#                 e0 = self.inOutEdges[t][0]
#                 e1 = self.inOutEdges[t][1]
#                 if e0 not in colorEdges :
#                     # colorEdges.append(e0)
#                     series.append(e0)
#                     series = _find_edge(series,e0)
#                 if e1 not in colorEdges :
#                     # colorEdges.append(e1)
#                     series.append(e1)
#                     series = _find_edge(series,e1)
#             return series

#         if not eid and not vid :
#             for v in self.nodes :
#                 if self.dgree[v] == 2 and v not in colorNodes:

#                     series = self.inOutEdges[v]
#                     colorNodes.append(v)
#                     colorEdges.extend(series)
#                     series = _find_edge(series,series[0])
#                     series = _find_edge(series,series[1])
#                     seriess.append(series)
        
#         # print(seriess)




#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   深度优先搜索通路
#     def GetPaths(self) :
#         return GetAllPaths(self.es, startNodes=None, targetNodes=None, giveways=None)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   最小树函数
#     def GetMinTree(self, weights=None) :
#         return _GetMinTree(self.es, weights=weights)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   独立回路函数
#     def GetCircuits(self, weights=None) :
#         return _GetCircuits(self.es, weights=weights)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   判断是否为连通图
#     def IsConnected(self) :
#         return self.dg.is_connected(mode=WEAK)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   提取联通块
#     def GetConnectedBlocks(self) :
#         """确定联通块
#         :返回联通块分支id列表    
#         """
#         cl = self.dg.components(mode=WEAK)
#         connectedBlocks = []
#         for g in cl.subgraphs():
#             # 子图的index与原图已经不一致了!
#             # 这里只能使用id属性进行返回!
#             connectedBlocks.append([e['id'] for e in g.es])
#         # 释放内存
#         del cl
#         return connectedBlocks

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   判断是否为 "非" 有向无环图(单向回路，包括单点回路loop)
#     def IsDag(self) :
#         return self.dg.is_dag()
#         #   检查图是否为DAG(有向无环图)。
#         #   DAG是没有有向环的有向图。
#         #   返回
#         #   boolean如果是DAG则为true，否则为False。

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   单节点环
#     def GetLoops(self) :
#         loops = [e["id"] for e in self.es if e["s"]==e["t"]]
#         return loops

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   单向回路
#     def GetUnidirectionalLoops(self) :
#         """查找含有单向回路的强联通块
#         :返回含有单向回路的强联通块
#         """
#         cl = self.dg.components(mode=STRONG)
#         comps = []
#         for g in cl.subgraphs():
#             # 至少2条边才能构成单向回路
#             if len(g.es) < 2:continue
#             # 子图的index与原图已经不一致了!
#             # 这里只能使用id属性进行返回!
#             comps.append([e["id"] for e in g.es])
#         # 释放内存
#         del cl
#         return comps

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   截流（用于密闭、盲巷处置）
#     def Closure(self, giveways=[]) :
#         # giveways 截流巷道、盲巷
#         flows       = list()                # 流通网络
#         targetNodes = self.sinkNodes        # 目标点, 汇点

#         def __Func(vid) :                   # 内嵌vid节点递归函数
#             boole = False                   # 搜索成功标识为假
#             if vid in targetNodes :         # 搜索节点为汇点
#                 boole = True                # 搜索成功标识为真
#                 return boole                # 递归返回

#             for eid in self.outEdges[vid] :         # 当前节点出边循环
#                 if eid not in giveways :            # 当前出边非避让
#                     giveways.append(str(eid))       # 当前出边加入避让

#                     if __Func(self.esDict[eid]["t"]) :      # 当前出边末节点调用成功
#                         flows.append(str(eid))              # 当前出边加入流通网络  ？？？盲巷？
#                         targetNodes.append(str(self.esDict[eid]["s"]))
#                         boole = True
            
#             return boole

#         for vid in self.sourceNodes :
#             __Func(vid)
 
#         return flows
       
# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
# #   搜索找单向回路，待论证
# def GetUCs(dg) :
#     ucs = []
#     #找全部单向回路
#     for e in dg.es :
#         #从分支末节点开始搜索到分支始节点
#         paths_index = dg.get_shortest_paths(e.target_vertex.index, e.source_vertex.index, 
#             weights=None, mode=OUT, output="epath")
#         if not len(paths_index[0]) :
#             continue
#         else :
#             path = GetIDEdge(dg,paths_index[0])

#         if not len(path) :
#             continue
#         else :
#             ucs.append( [e["id"]] + path)
#     #合并分支相同只是排列次序不同的重复单向回路
#     ucs1 = []
#     for uc in ucs :
#         c = set(uc)
#         if c not in ucs1 :
#             ucs1.append(c)
#     # return ucs1

#     #确定无通路分支================有误
#     #ucEdges = []

#     #源汇节点
#     sourceNodes = dg.vs.select(_indegree=0)
#     sinkNodes = dg.vs.select(_outdegree=0)


#     hasPath = set()
#     noPath = set()
#     #单向回路循环
#     for uc in ucs1 :
#         #单向回路分支循环
#         for eid in uc :
#             #print("eid===============",eid)
#             index = GetIndexEdge(dg, eid)       #当前分支dg索引
#             e = dg.es[index]                    #当前分支dg对象
#             #分支末节点到所有汇点正向通路
#             for sinkNode in sinkNodes :
#                 sinkPath = dg.get_shortest_paths(
#                 e.target_vertex.index, sinkNode, weights=None, mode=OUT, output="epath")[0]

#                 #分支始节点到所有源点逆向通路
#                 for sourceNode in sourceNodes :
#                     sourcePath = dg.get_shortest_paths(
#                     e.source_vertex.index, sourceNode, weights=None, mode=IN, output="epath")[0]

#                     if len(sinkPath) != 0 and len(sourcePath)!= 0 :

#                         if not len((set(sinkPath) & set(sourcePath))) :
#                             hasPath.add(eid)
#                         else :
#                             noPath.add(eid)
#     noPath -= (noPath & hasPath)
#     # print(hasPath)
#     # print(noPath)
#     return ucs1,[eid for eid in noPath]






# #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
# #
# #   网络解算拓扑检查
# #
# def CheckNetNC(edges) :
#     # 1 连通性检查
#     if not IsConnected(edges) :                         # 非连通图
#         blocks = GetConnectedBlocks(edges)
#         raise __MyAppValueError(
#             message =   ERROR_MESS_NO_CONNECTED,
#             code    =   ERROR_CODE_NO_CONNECTED,       # code=10201,
#             data    =   {DATA : blocks}
#         )

#     # 2 单节点环检查（单向回路不检查）
#     loops = GetLoops(edges)                 # 环集
#     if len(loops) > 0 :                     # 有单节点环
#         raise __MyAppValueError(
#             message =   ERROR_MESS_RING,      # 出风井口共享
#             code    =   ERROR_CODE_RING,         # 10207
#             data    =   {DATA : loops}                
#         ) 

#     # 3 源点、汇点、源支、汇支检查
#     # 3.1 无进风井
#     net = CreateNetwork(edges)
#     if len(net[SOURCE_NODES]) == 0 :                        # 源点数为0
#         raise __MyAppValueError(
#             message =   ERROR_MESS_NO_SHAFTIN,              # 无进风井
#             code    =   ERROR_CODE_NO_SHAFTIN,              # 10202
#             data    =   {DATA : None}
#         )

#     # 3.2 无出风井
#     if len(net[SINK_NODES]) == 0 :                          # 汇点数为0
#         raise __MyAppValueError(
#             message =   ERROR_MESS_NO_SHAFTOUT,             # 无出风井
#             code    =   ERROR_CODE_NO_SHAFTOUT,             # 10203
#             data    =   {DATA : None}
#         )        

#     # 3.3 进风井口共享
#     if len(net[SOURCE_EDGES]) > len(net[SOURCE_NODES]) :    # 源支数大于源点数
#         raise __MyAppValueError(
#             message =   ERROR_MESS_SHARE_ENTRANCE,          # 进风井口共享
#             code    =   ERROR_CODE_SHARE_EXIT,              # 10204
#             data    =   {DATA : net[SOURCE_EDGES]}                
#         ) 

#     # 3.4 回风井口共享
#     if len(net[SINK_EDGES]) > len(net[SINK_NODES]) :        # 汇支数大于汇点数
#         raise __MyAppValueError(
#             message =   ERROR_MESS_SHARE_EXIT,                     # 出风井口共享
#             code    =   ERROR_CODE_SHARE_EXIT,                   # 10205
#             data    =   {DATA : net[SINK_EDGES]}                
#         ) 

#     return True
# #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

# # igraph 最长路慢
# # 曹鹏的快，但是不能有单向回路
# def GetLongestPath(edges, start, target, weights=None, maxWeight=10000) :
#     # print("edges=",edges)
#     dg, edgeIdIndexMap, nodeIdIndexMap = GetDGraph(edges)
#     weights_ = [-weights[e[ID]] for e in dg.es]
#     print(weights_)
#     s = nodeIdIndexMap[start]
#     t = nodeIdIndexMap[target]
#     # print(s,t)
#     path = dg.get_shortest_paths(v=s, to=t, weights=weights_, mode='out', output='epath')
#     print(path)
#     path_ = [dg.es[i][ID] for i in path[0]]
#     return path_



# #   id-index保持一致
# def GetShortestPath(edges, start, target, weights=None) :
#     dg, edgeIdIndexMap, nodeIdIndexMap = GetDGraph(edges)
#     weights_ = [weights[e[ID]] for e in dg.es]
#     s = nodeIdIndexMap[start]
#     t = nodeIdIndexMap[target]

#     path = dg.get_shortest_paths(v=s, to=t, weights=weights_, mode='out', output='epath')
#     print(path)
#     path_ = [dg.es[i][ID] for i in path[0]]
#     return path_

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

