#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	矿井通风网络图类
#
###########################################################################################################################


from    jl.network.i_graph.graph    import  G
from    cloud.jl.network.get_topology         import  get_topology    as  _get_topology_

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   截流（用于密闭、盲巷处置）                  暂未使用JianGraph的深度优先搜索
def closure(edges, giveways=[]) :
        # giveways 截流巷道、盲巷
        flows       = list()                    # 流通网络
        g = G()
        top = _get_topology_(edges=edges)
        # top = get_topology(edges)
        targetNodes = top['sinkNodes']          # 目标点（一般为汇点，含盲巷）
        sourceNodes = top['sourceNodes']
        # nodeOutEdges = top['nodeOutEdges']
        nodeOutEdges = g.get_node_out_edge(edges=edges)

        mape = top['mape']

        def _Func_(vid) :                   # 内嵌vid节点递归函数
            boole = False                   # 搜索成功标识为假
            if vid in targetNodes :         # 搜索节点为汇点
                boole = True                # 搜索成功标识为真
                return boole                # 递归返回

            for eid in nodeOutEdges[vid] :         # 当前节点出边循环
                if eid not in giveways :            # 当前出边非避让
                    giveways.append(eid)       # 当前出边加入避让

                    if _Func_(top['mape'][eid]["t"]) :      # 当前出边末节点调用成功
                        flows.append(eid)              # 当前出边加入流通网络  ？？？盲巷？
                        targetNodes.append(top['mape'][eid]["s"])
                        boole = True
            
            return boole

        for vid in sourceNodes :
            _Func_(vid)
 
        return flows

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



            


    
        









#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#         # 5. 闭合处理虚拟分支
#         net.closed = closed
#         if net.closed :
#         	# 5.1 基点-源点虚拟分支
#             i = 1
#             net.es_virtual_source = list()
#             for vid in net.sourceNodes :               # 基点-源点循环
#                 #创建虚拟源支
#                 edge = {"id":'virtual_source_%d' % i,"s":"virtual","t":vid}
#                 net.es_virtual_source.append(edge)     
#                 i += 1
#             # 5.2 汇点-基点虚拟分支
#             i = 1
#             net.es_virtual_sink = list()            # 汇点-基点虚拟分支
#             for vid in net.sinkNodes :
#                 edge = {"id":'virtual_sink_%d' % i,"s":vid,"t":"virtual"}
#                 net.es_virtual_sink.append(edge)
#                 i += 1
#             # 5.3 全部虚拟分支
#             net.es_virtual_all = list()
#             net.es_virtual_all.extend(net.es_virtual_source)
#             net.es_virtual_all.extend(net.es_virtual_sink)

#         else :
#             net.es_virtual_all     =   []
#             net.es_virtual_source  =   []
#             net.es_virtual_sink    =   []

#         # 6. igraph
#         net.dg, net.edgeIdIndexMap, net.nodeIdIndexMap = _GetDGraph(net.es)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   搜索串联风路
#     def SearchSeriesEdges(net, eid=None,vid=None,graph="UNDIRECTED") :
#         seriess = list()
#         colorNodes = list()
#         colorEdges = list()

#         def _find_edge(series,eid) :
#             colorEdges.append(eid)
#             edge = net.esDict[eid]
#             s,t = edge["s"], edge["t"]
#             if s not in colorNodes and net.dgree[s]==2 :
#                 colorNodes.append(s)
#                 e0 = net.inOutEdges[s][0]
#                 e1 = net.inOutEdges[s][1]
#                 if e0 not in colorEdges :
#                     # colorEdges.append(e0)
#                     series.insert(0,e0)
#                     series = _find_edge(series,e0)
#                 if e1 not in colorEdges :
#                     # colorEdges.append(e1)
#                     series = series.insert(0,e1)
#                     _find_edge(series,e1)
#             if t not in colorNodes and net.dgree[t]==2 :
#                 colorNodes.append(t)
#                 e0 = net.inOutEdges[t][0]
#                 e1 = net.inOutEdges[t][1]
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
#             for v in net.nodes :
#                 if net.dgree[v] == 2 and v not in colorNodes:

#                     series = net.inOutEdges[v]
#                     colorNodes.append(v)
#                     colorEdges.extend(series)
#                     series = _find_edge(series,series[0])
#                     series = _find_edge(series,series[1])
#                     seriess.append(series)
        
#         # print(seriess)




#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   深度优先搜索通路
#     def GetPaths(net) :
#         return GetAllPaths(net.es, startNodes=None, targetNodes=None, giveways=None)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   最小树函数
#     def GetMinTree(net, weights=None) :
#         return _GetMinTree(net.es, weights=weights)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   独立回路函数
#     def GetCircuits(net, weights=None) :
#         return _GetCircuits(net.es, weights=weights)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   判断是否为连通图
#     def IsConnected(net) :
#         return net.dg.is_connected(mode=WEAK)

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   提取联通块
#     def GetConnectedBlocks(net) :
#         """确定联通块
#         :返回联通块分支id列表    
#         """
#         cl = net.dg.components(mode=WEAK)
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
#     def IsDag(net) :
#         return net.dg.is_dag()
#         #   检查图是否为DAG(有向无环图)。
#         #   DAG是没有有向环的有向图。
#         #   返回
#         #   boolean如果是DAG则为true，否则为False。

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   单节点环
#     def GetLoops(net) :
#         loops = [e["id"] for e in net.es if e["s"]==e["t"]]
#         return loops

#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   单向回路
#     def GetUnidirectionalLoops(net) :
#         """查找含有单向回路的强联通块
#         :返回含有单向回路的强联通块
#         """
#         cl = net.dg.components(mode=STRONG)
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
#     def Closure(net, giveways=[]) :
#         # giveways 截流巷道、盲巷
#         flows       = list()                # 流通网络
#         targetNodes = net.sinkNodes        # 目标点, 汇点

#         def __Func(vid) :                   # 内嵌vid节点递归函数
#             boole = False                   # 搜索成功标识为假
#             if vid in targetNodes :         # 搜索节点为汇点
#                 boole = True                # 搜索成功标识为真
#                 return boole                # 递归返回

#             for eid in net.outEdges[vid] :         # 当前节点出边循环
#                 if eid not in giveways :            # 当前出边非避让
#                     giveways.append(str(eid))       # 当前出边加入避让

#                     if __Func(net.esDict[eid]["t"]) :      # 当前出边末节点调用成功
#                         flows.append(str(eid))              # 当前出边加入流通网络  ？？？盲巷？
#                         targetNodes.append(str(net.esDict[eid]["s"]))
#                         boole = True
            
#             return boole

#         for vid in net.sourceNodes :
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

