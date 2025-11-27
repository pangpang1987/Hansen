#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	极值通路模块
#
###########################################################################################################################

#   1. 功能: 计算节点间最长路和最短路
#   2. 说明: 
#   （1）网络必须为正向网络
#   （2）负风量分支需调整始末节点
#   （3）权重可以为负值, 但不能视为分支反向
#   （4）可以处理并联分支
#   （5）不能含有单向回路，否则死循环
#   3. 数据结构
#   3.1 输入数据结构
#   （1）拓扑关系
#   （2）分支权重
#   3.2 输出数据结构
#   （1）所有汇点的最长路
#   （2）全网最长路
#   （3）所有汇点最短路
#   （4）全网最短路
"""
{
    'maxPaths': [       # 汇点最长通路
        {'s': 'v2', 't': 'v7', 'maxPath': ['e2', 'e5', 'e7', 'e9'], 'weight': 4}, 
        {'s': 'v2', 't': 'v22', 'maxPath': ['e2', 'e5', 'e7', 'e9', 'e22'], 'weight': 14}
    ], 
    'maxPath': {'s': 'v2', 't': 'v22', 'maxPath': ['e2', 'e5', 'e7', 'e9', 'e22'], 'weight': 14}, 
    'minPaths': [
        {'s': 'v2', 't': 'v7', 'minPath': ['e2', 'e4'], 'weight': 2}, 
        {'s': 'v2', 't': 'v22', 'minPath': ['e2', 'e4', 'e22'], 'weight': 12}
    ], 
    'minPath': {'s': 'v2', 't': 'v7', 'minPath': ['e2', 'e4'], 'weight': 2}
}
"""

from    jl.jGraph.jGraph    import  JGraph
from    cloud.jl.common.dataProcessing.globalVariable   import  *
import math

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   提取极值通路函数
#
def GetPathsExtreme(
        edges,          # 分支列表, [{"id":id,"s":s,"t":t,"weight":weight}]
        fromNodes=[],   # 起始点id列表, [vid], 如果列表为空, 则为网络全部源点
        toNodes=[],     # 目标点id列表, [vid], 如果列表为空, 则为网络全部汇点
        extreme='ALL'     # 极值类型, "all":最大和最小; "max":仅最大; "min":仅最小
):

    jG = JGraph(edges)

    if not fromNodes :
        fromNodes = jG.sourceNodes          # 网络源点列表

    if not toNodes :
        toNodes = jG.sinkNodes              # 网络汇点列表

    ss = jG.vsStart                         # 风路始节点列表
    ts = jG.vsTarget                        # 风路末节点列表

    return _GetExtremePaths(
        edges,
        fromNodes,
        toNodes,
        ss,             # 分支始节点id列表
        ts,             # 分支末节点id列表
        extreme
    )
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
def _GetExtremePaths(edges,fromNodes,toNodes,ss,ts,extreme=ALL) :
    all_paths = dict()
    max_paths_cl = list()
    min_paths_cl = list()
    for v in fromNodes:
        min_paths = w_paths(edges,v,toNodes,ts)
        max_paths = w_paths(edges,v,toNodes,ts,max_path=True)
        for end_node in min_paths:
            dict_1= dict()
            dict_1['s'] = v
            dict_1['t'] = end_node
            dict_1['minPath'] = min_paths[end_node][0]
            dict_1['weight'] = min_paths[end_node][1]
            min_paths_cl.append(dict_1)
        for end_node in max_paths:
            dict_2= dict()
            dict_2['s'] = v
            dict_2['t'] = end_node
            dict_2['maxPath'] = max_paths[end_node][0]
            dict_2['weight'] = max_paths[end_node][1]
            max_paths_cl.append(dict_2)
    all_paths['maxPaths'] = sorted(max_paths_cl,key = lambda i: i['weight'])
    all_paths['maxPath'] = all_paths['maxPaths'][-1]
    all_paths['minPaths'] = sorted(min_paths_cl,key = lambda i: i['weight'])
    all_paths['minPath'] = all_paths['minPaths'][0]
    
    if extreme == 'MIN':            # 最短路
        return {'minPaths': all_paths['minPaths'],'minPath': all_paths['minPath']}
    if extreme == 'MAX':            # 最长路
        return {'maxPaths': all_paths['maxPaths'],'maxPath': all_paths['maxPath']}

    if extreme == 'ALL':            # 最短路和最长路
        return all_paths
        
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   计算通路权重
#
def w_paths(edges, start, ends, ts, max_path = False):

    """
    使用 Dijkstra 算法计算指定点 start到任意点的最短路径/最长路径的距离
    edges: 分支列表[{edge1},{edges2},...]
    start: 字符，起始点
    end: 列表[]，终点
    max_path: 控制输出最短或最长路径，max_path=True 输出最长路径
    return：返回 最长/最短 路径分支列表
    """

    # 权重字段名
    weight_name = 'weight'

    # 最短路链接表
    graph = graph_edge(edges,weight_name,P=True,P_min=True)

    # 最长路链接表
    if max_path == True:
        graph = graph_edge(edges,weight_name,P=True,P_max=True) 

    # T = find_T(edges)
    # for t in T:
    for t in ts :       # 分支末节点id列表
        if t not in graph:
            graph[t]={} # 整理为链接表格式

    
    sum_weight = {}  # start到其他节点的权重之和
    paths = {start:[start]} # 路径
 
    # 初始化sum_weights
    for key in graph.keys():
        if max_path == False:
            sum_weight[key] = math.inf # 初始化最短路sun_weights
        if max_path == True:
            sum_weight[key] = -math.inf # 初始化最长路sun_weights

    sum_weight[start] = 0 # 始节点为0
    queue = [start] # 创建队列
    while len(queue) != 0: #循环队列
        s = queue[0] # 起始节点
        for key in graph[s].keys(): # 遍历子节点
            dis = graph[s][key] + sum_weight[s] # 更新距离
            if ((max_path == False and sum_weight[key] > dis) 
                or (max_path == True and sum_weight[key] < dis)): # 判断距离是否增长或减短
                sum_weight[key] = dis # 更新权重
                # temp = list_copy.main(paths[s])
                temp = paths[s][:]
                temp.append(key)        
                paths[key] = temp # 最优路径为起始节点最优路径+key
                queue.append(key)
        queue.pop(0)  # 删除原来的起始节点

    # 输出分支id
    id_dict = {}
    for end in ends:
        edge_record = {} # 分支整理
        if end not in paths:
            if max_path == True:
                id_dict[end] = (None,-math.inf)
            else:
                id_dict[end] = (None,math.inf)
        else:
            s_t = paths[end] 
            weight = sum_weight[end]
            for id,vertex in enumerate(s_t):
                for edge in edges:
                    if id+1 <= len(s_t)-1:
                        if edge['s'] == vertex and edge['t'] == s_t[id+1]:
                            if f'{vertex}-{s_t[id+1]}' not in edge_record: # 添加已检索的分支
                                edge_record[f'{vertex}-{s_t[id+1]}'] = [edge[weight_name],edge['id']]
                            if max_path == False:
                                if f'{vertex}-{s_t[id+1]}' in edge_record and edge_record[f'{vertex}-{s_t[id+1]}'][0]>edge[weight_name]:
                                    edge_record[f'{vertex}-{s_t[id+1]}'] = [edge[weight_name],edge['id']]
                            if max_path == True:
                                if f'{vertex}-{s_t[id+1]}' in edge_record and edge_record[f'{vertex}-{s_t[id+1]}'][0]<edge[weight_name]:
                                    edge_record[f'{vertex}-{s_t[id+1]}'] = [edge[weight_name],edge['id']]
            # 输出分支
            edges_output = []
            for edge in edge_record.values():
                edges_output.append(edge[1])
            if max_path == True:
                id_dict[end] = (edges_output,weight)
            else:
                id_dict[end] = (edges_output,weight)
    return id_dict
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥

#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
def graph_edge(edges,weight_name,P=False,P_min=False,P_max=False):
    '''
    edges: 分支列表，[{edge1},{edge2},...]
    P: 布尔值，输出带权重/无权重 链接表
    P_min: 控制并联巷道输出，P_min=True输出并联巷道中权值小的值
    P_max: 控制并联巷道输出，P_max=True输出并联巷道中权值大的值
    输出邻接表
    '''
    # 找到每条边的源点和汇点
    s_nodes = list()
    for i in edges:
        if i['s'] not in s_nodes:
            s_nodes.append(i['s'])
    t_nodes = list()
    for i in edges:
        if i['t'] not in t_nodes:
            t_nodes.append(i['t'])

    # 输出不带权重链表
    s_t = dict()
    for s in s_nodes:
        t = list()
        for edge in edges:
            if s == edge['s']:
                t.append(edge['t'])
        s_t[s] = t

    # 带权重链表
    m_t = {}
    for s in s_t:
        link_ = {}
        for t in s_t[s]:
            weight = []
            for edge in edges:
                if edge['s'] == s and edge['t'] == t:
                    weight.append(edge[weight_name])
            if len(weight)==1:
                link_[t] = weight[0]
            elif P_min == True:
                link_[t] = min(weight)
            elif P_max == True:
                link_[t] = max(weight)
            else:
                link_[t] = weight
        m_t[s] = link_
    if P==False:
        return s_t   
    if P==True:
        return m_t
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥




