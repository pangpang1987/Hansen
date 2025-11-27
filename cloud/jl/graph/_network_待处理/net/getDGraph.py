#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#	JGraph	图类模块
#
######################################################################################


from igraph  import  *

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建 igraph 对象
#
#   创建图的拓扑关系(节点和分支,以及实际id与igraph内部编号之间的映射)
def GetDGraph(edges) :
    """创建igraph对象函数
    :edges 拓扑列表
    :返回参数----(dg,edgeIdIndexMap,nodeIdIndexMap)
    """
    dg = Graph(directed=True)
    #收集所有的节点
    sIDs = [e["s"] for e in edges]
    tIDs = [e["t"] for e in edges]
    sIDs.extend(tIDs)
    #构造节点集合(利用set去除重复编号,并排序,然后再转换成list)
    nIDs = set(sIDs)
    #构造节点编号到igraph内部编号的映射关系
    nodeIdIndexMap = dict(zip(nIDs, range(len(nIDs))))
    #有向图增加节点(igraph节点内部编号从0开始)
    dg.add_vertices(len(nodeIdIndexMap))

    #构造分支编号到igraph内部编号的映射关系
    edgeIdIndexMap = {}
    #分支的实际编号
    e_index = 0
    for e in edges:
        # eID, u, v = str(e['id']), str(e['s']), str(e['t'])
        eID, u, v = e["id"], e["s"], e["t"]
        u, v = nodeIdIndexMap[u], nodeIdIndexMap[v] # 转换成igraph内部的编号
        if eID not in edgeIdIndexMap:            # 已经考虑并联边!!!
            dg.add_edge(u, v)  #增加新分支(igraph分支内部编号从0开始)
            edgeIdIndexMap[eID] = e_index #记录分支编号到igraph内部编号的映射关系
            e_index += 1
    #设置igraph节点的id属性数据
    for id,index in nodeIdIndexMap.items() :
        dg.vs[index]["id"] = id
    #设置igraph分支的id属性数据
    for id,index in edgeIdIndexMap.items() :
        dg.es[index]["id"] = id
    return dg, edgeIdIndexMap, nodeIdIndexMap
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   GetPaths
#
def _GetPaths(dg,nodeMap,starts=None,targets=None) :
    if not starts :
        startIndexs = [v.index for v in dg.vs.select(_indegree=0)]
    else :
        startIndexs = [nodeMap[vid] for vid in starts]

    if not targets :
        targetIndexs = [v.index for v in dg.vs.select(_outdegree=0)]
    else :
        targetIndexs = [nodeMap[vid] for vid in targets]
        
    eidPaths = []
    for v in startIndexs :
        paths = dg.get_all_simple_paths(v,targetIndexs, cutoff=-1, mode=OUT)
        for p in paths :
            path = node_path_to_edge_path(dg, p, directed=False)
            eidPath = [dg.es[i]["id"] for i in path]
            eidPaths.append(eidPath)
    return eidPaths
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥







#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   根据节点id查找dg的内部编号index
#
def GetIndexNode(dg, node_id) :
    for v in dg.vs :
        if v["id"] == node_id :
            return v.index
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
# 
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   根据节点dg索引提取节点ID
#
def GetIDNode(dg, node_index) :
    return dg.vs[node_index]["id"]

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   根据分支id查找分支的内部编号
#
def GetIndexEdge(dg, edge_id) :
    for e in dg.es :
        if e["id"] == edge_id :
            return e.index
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   根据分支索引提取分支ID
#           错误
def GetIDEdge(dg, edge_index) :
    e = dg.es[edge_index]
    return e["id"]
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   提取节点集合
#
def GetNodes(es) :
    vs,vs_map = [],{}
    for e in es :
        vs_map[e.start.id] = e.start
        vs_map[e.target.id] = e.target
    return ([v for k,v in vs_map.items()],vs_map)
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   两点间最短路
#
#   计算图中 from/to 一个给定节点的最短路径。


def GetShortestPaths(dg,v,to=None,weights=None, mode=OUT, output="epath") :
    """两点间通路
    参数:
    v ----计算路径的 源/目标
    to----描述计算路径的 目标/源 的顶点选择器。这可以是单个顶点ID，一个顶点ID列表，一个顶点名称列表，
    一个顶点名称列表或一个VertexSeq对象。None表示所有的顶点。
    weights----列表中的边权值或包含边权值的边属性的名称。如果没有，则假定所有边的权值相等。
    mode----路径的方向性。IN表示计算进来的路径，OUT表示计算出去的路径，ALL表示同时计算这两个路径。
    output----确定应该返回什么。如果这是“vpath”，将返回一个顶点id列表，每个目标顶点都有一个路径。
    对于未连接图，一些列表元素可能是空的。注意，在mode=in的情况下，路径中的顶点将以相反的顺序返回。
    如果output="epath"，返回的是边缘id而不是顶点id。
    返回:
    请参阅输出参数的文档。
    """
    startIndex  = GetIndexNode(dg, v)
    if to == None :
        targetIndex = None
    else :
        targetIndex = GetIndexNode(dg, to)
    return GetIDEdge(dg,dg.get_shortest_paths(
        startIndex, targetIndex, weights, mode, output)[0])
#
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   计算图中从一个给定节点到其他节点(或所有节点)的所有简单路径(节点)
#
#   说明：
# 如果一个路径的顶点是唯一的，那么它就是简单的，也就是说，每个顶点都不会被访问超过一次。
# 注意，在一个图的两个顶点之间可能存在指数级的路径，特别是如果你的图是格状的。
# 在这种情况下，您可能会在使用这个函数时耗尽内存。
# def GetPaths(edges,starts=None,targets=None) :


#     pass
def GetAllSimplePathsNode(dg,v,to=None, cutoff=-1, mode=OUT) :
    """

    """
    startIndex  = GetIndexNode(dg, v)
    if to == None :
        targetIndex = None
    else :
        targetIndex = GetIndexNode(dg, to)
    
    ps = dg.get_all_simple_paths(startIndex, targetIndex, cutoff, mode)
    paths = []
    for path in ps :
        paths.append([GetIDNode(dg,i) for i in path])
    return paths

    #return  dg.get_all_simple_paths(startIndex, targetIndex, cutoff, mode)


#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
# 节点路径转换成分支路径
def node_path_to_edge_path(dg, node_path, directed=False):
    if len(node_path) < 2:
        return []
    # print(node_path)
    edge_path = []
    for u,v in zip(node_path[:-1], node_path[1:]):
        i = dg.get_eid(u, v, directed=directed, error=False)
        if i < 0:
            continue
        else:
            edge_path.append(i)
    return edge_path
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求最小树
#
def _GetMinTree(dg,weights=None, return_tree=False) :
    """计算一个图的最小生成树
    参数:
    weights----一个包含图中每条边的权值的向量。None表示图是未加权的。
    return_tree ----是返回最小生成树(当return_tree为True时)还是返回最小生成树的边id
    (当return_tree为False时)。由于历史原因，默认为True，因为这个参数是在igraph 0.6中引入的。
    返回:
    如果return_tree为真，则生成树作为图形对象;如果return_tree为假，则生成树的边缘id为图形对象。
    """
    return [dg.es[i]["id"] for i in dg.spanning_tree(weights,return_tree)]
#

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求最小树
#
def GetMinTree(top,weights=None) :
    dg,map_edge,map_node = GetDGraph(top)
    if weights :
        weights = _GetSetDGWeight(dg,weights)
    return _GetMinTree(dg,weights=weights,return_tree=False)
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   给 dg 分支赋权重
#
def _GetSetDGWeight(dg,weights) :
    """给dg分支赋权重
    :dg----igraph对象
    :weights----分值权重词典 {"eid" : w}
    :weights----返回值列表，与dg index一致
    """
    for e in dg.es :
        e["weight"] = weights[e["id"]]
    return [e["weight"] for e in dg.es]
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥



#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求回路函数
#
def GetCircuits(topList,weights=None) :
    """求回路函数
    :top----拓扑列表
    :weights----分支权重
    :cs----返回回路列表 [[{"d":d,"eid":eid}]]
    """
    minTree = GetMinTree(topList,weights)

    tree = []
    cotree = []
    map = {}
    for e in topList :
        map[e["id"]] = e
        if e["id"] in minTree :
            tree.append(e)
        else :
            cotree.append(e)
    dg,map1,map2 = GetDGraph(tree)

    cs = []
    for e in cotree:
        path = GetShortestPaths(dg,e["t"],e["s"],weights=None, mode=ALL, output="epath")
        c=[]
        c.append({"d":1,"eid":e["id"]})
        node = e["t"]
        for eid in path :
            edge = map[eid]
            if edge["s"] == node :
                c.append({"d":1,"eid":eid})
                node = edge["t"]
            else :
                c.append({"d":-1,"eid":eid})
                node = edge["s"]
        cs.append(c)
    return cs
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€



    
# gomory_hu_tree(capacity=None, flow="flow")
# maxflow(source, target, capacity=None)
# mincut(source=None, target=None, capacity=None)
# st_mincut(source, target, capacity=None)
# path_length_hist(directed=True)
# gomory_hu_tree(capacity=None, flow="flow")
# # tree
# spanning_tree

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	构造JNetwork
# def f20200217(ss):
#     # 表头，必须唯一，所以一个空为有空格，一个空为没有空格，效果一样
#     col_name = [u"入边","", u"节点", " ",u"出边"]
#     # 创建表,有表格绘制线
#     # hrule 参数
#         # 0 FRAME 表示水平表格横线只有框架，三线表，默认
#         # 1 ALL   表示打印所有横线
#         # 2 NONE  表示没有横线
#         # 3 HEADER 表示只有表头
#     # vrules 参数
#         # 0 FRAME 表示表格竖线只有框架
#         # 1 ALL   表示打印所有竖线,默认
#         # 2 NONE  表示没有竖线
#         # 3 HEADER 表示只有表头
#     # 无绘制线表格, hrules=2, vrules=2
#     # 如果要打印三线表，删除hrules，vrules两个参数，使用默认的就行
#     # 如果要打印表格的所有线，hrules=1,vrules=1（或者这个参数不用设置）

#     # 1.创建无线条的表格
#     # x = PrettyTable(col_name,hrules=2,vrules=2,encoding = "gbk")

#     # 2. 创建三线表
#     x = PrettyTable(col_name,encoding = "gbk")

#     # 3. 创建所有线条的表格
#     x = PrettyTable(col_name,hrules=1,encoding = "gbk")

#     # 以 '节点' 这列排序
#     x.sortby = u"节点"
#     # 入边这列左对齐,默认居中，'c'
#     x.align[u"入边"] = "l"
#     # 出边右对齐,默认居中，'c'
#     x.align[u"出边"] = "r"
#     # 假设的拓扑，正常循环在这
#     for s in ss:
#     	x.add_row(s)
#     #x.add_row([u'无','-->','v0','-->','e1'])
#     #x.add_row(['e1 e2','-->','v1','-->','e3'])
#     #x.add_row(['','-->','v2','-->','e1 e3 e12345'])
#     print(x)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# Python实现BFS


graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

aaa=list(bfs_paths(graph, 'A', 'F'))
print(aaa)



#	结果：[['A', 'C', 'F'], ['A', 'B', 'E', 'F']]

#Python实现DFS

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

x1=list(dfs_paths(graph, 'A', 'F'))
print(x1)

#[['A', 'C', 'F'], ['A', 'B', 'E', 'F']]









1 # 图的广度优先遍历
 2 # 1.利用队列实现
 3 # 2.从源节点开始依次按照宽度进队列，然后弹出
 4 # 3.每弹出一个节点，就把该节点所有没有进过队列的邻接点放入队列
 5 # 4.直到队列变空
 6 from queue import Queue
 7 def bfs(node):
 8     if node is None:
 9         return
10     queue = Queue()
11     nodeSet = set()
12     queue.put(node)
13     nodeSet.add(node)
14     while not queue.empty():
15         cur = queue.get()               # 弹出元素
16         print(cur.value)                # 打印元素值
17         for next in cur.nexts:          # 遍历元素的邻接节点
18             if next not in nodeSet:     # 若邻接节点没有入过队，加入队列并登记
19                 nodeSet.add(next)
20                 queue.put(next)



1 # 图的深度优先遍历
 2 # 1.利用栈实现
 3 # 2.从源节点开始把节点按照深度放入栈，然后弹出
 4 # 3.每弹出一个点，把该节点下一个没有进过栈的邻接点放入栈
 5 # 4.直到栈变空
 6 def dfs(node):
 7     if node is None:
 8         return
 9     nodeSet = set()
10     stack = []
11     print(node.value)
12     nodeSet.add(node)
13     stack.append(node)
14     while len(stack) > 0:
15         cur = stack.pop()               # 弹出最近入栈的节点
16         for next in cur.nexts:         # 遍历该节点的邻接节点
17             if next not in nodeSet:    # 如果邻接节点不重复
18                 stack.append(cur)       # 把节点压入
19                 stack.append(next)      # 把邻接节点压入
20                 set.add(next)           # 登记节点
21                 print(next.value)       # 打印节点值
22                 break                   # 退出，保持深度优先

"""



