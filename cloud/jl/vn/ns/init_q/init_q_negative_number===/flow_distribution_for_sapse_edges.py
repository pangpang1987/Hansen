from collections import deque
from collections import Counter
import math

# 无穷大
INF = float('inf')


def is_l(l):
    '''
    判断是否有下界
    @l：下界'''
    if l:
        return True


def target_list(edge_list, t=None):
    '''
    数据格式[{},{}]
    输出数据列表
    '''
    S = list()
    for edge in edge_list:
        S.append(edge[t])
    return S


def find_S(edges_list):
    '''
    用来找出源点
    :return: 源点编号
    '''
    # 读取起点列数据
    S = list()
    for s in target_list(edges_list, t='s'):
        if s not in target_list(edges_list, t='t'):
            S.append(s)
    return list(set(S))


def find_T(edges_list):
    '''
    用来找出汇点
    :return: 汇点编号
    '''
    # 读取起点列数据
    T = list()
    for s in target_list(edges_list, t='t'):
        if s not in target_list(edges_list, t='s'):
            T.append(s)
    return list(set(T))


def mark(_list):
    '''
    将字符串修改借口调用数据类型（字符串修改为数字类型）
    list:字符串列表
    return：返回字符串对应的id字典
    '''
    # 影响速度
    list1 = list()
    for i in _list:
        if i not in list1:
            list1.append(i)

    # 创建映射字典
    dict_str2id = dict()
    for i, j in enumerate(list1):
        dict_str2id[j] = i + 1

    return dict_str2id


def give_id(list_, dict_):
    '''
    赋予字符串id
    '''
    for i, j in enumerate(list_):
        for k in dict_:
            if j == k:
                list_[i] = dict_[k]
    return list_


def output_is_l(l, ans, edges_list):
    '''
    输出可行流
    @l：下界
    @ans：Dinic计算结果
    return：输出字典格式数据
    '''
    # 分配的流量需加上下界
    for i in range(len(l)):
        ans[1][i][2] += l[i]
        del ans[1][i][3]
        ans[1][i].insert(0, target_list(edges_list, t='id')[i])

    # 整理数据
    ans = ans[1][:len(l)]
    for i, j in enumerate(edges_list):
        ans[i][1] = edges_list[i]['s']
        ans[i][2] = edges_list[i]['t']

    # 输出格式{'edges':[{'id':, 's':, 't', 'q'}, , ...]}
    output = dict()
    output['edges'] = list()
    index = ['id', 's', 't', 'Q']

    for i in range(len(ans)):
        zip_1 = dict(zip(index, ans[i]))
        output['edges'].append(zip_1)
    return output


def input(edges_dict, low_l_1):
    '''
    分支数据
    {'edges':[
                {},
                ...
                {}
             ]
    }

    返回：以字典格式输出可行流分配
    '''
    # 读取数据
    edges = list()
    L = list()
    sum1 = Counter()
    sum2 = Counter()

    # 获取edges数据
    edges_list = edges_dict

    # 检验源汇点是否为1个
    # 源汇点集合
    S_list = find_S(edges_list)
    T_list = find_T(edges_list)

    # 读取总节点列表
    s_list = target_list(edges_list, t='s')
    t_list = target_list(edges_list, t='t')
    total_list = s_list + t_list

    # 数字类型
    node_max = len(set(s_list)) + len(set(t_list))

    # s_list 字符串转换为唯一的制定数字
    mark_ = mark(total_list)
    # s_list 字符串转换为唯一的制定数字
    s_list = give_id(s_list, mark_)
    t_list = give_id(t_list, mark_)
    S_list = give_id(S_list, mark_)
    T_list = give_id(T_list, mark_)

    # 读取数据
    for i in range(len(s_list)):
        # 起点
        a = int(s_list[i])
        # 止点
        b = int(t_list[i])
        # 容量下界
        l = target_list(edges_list, t='fixedQ')[i]
        # 考虑漏风qq
        qq = target_list(edges_list, t='sourceSinkQ')[i]
        # 如果没固定下界，下界为low_l_1，不设上界
        if l == None and qq != None:
            l = -qq + low_l_1
            u = INF
        elif l == None and qq == None:
            l = low_l_1
            u = INF
        else:
            l = l
            u = l
        edges.append((a, b, u - l))
        L.append(l)
        sum1[b] += l
        sum2[a] += l
    # 记录添加单源汇前的边数
    len_ori = len(edges)
    # 如果不是单源汇点，增加单个源汇点
    # 不是单源，单汇，添加单源点和单汇点及边，且节点数+2
    # 添加源汇
    S = int(node_max + 1)
    T = int(node_max + 2)
    # 更新节点数
    node_max = node_max + 2
    # 节点数
    n = int(node_max)
    # 添加统一的源汇
    for s in S_list:
        edges.append((S, int(s), INF))
        sum1[s] += 0
        sum2[S] += 0
    for t in T_list:
        edges.append((int(t), T, INF))
        sum1[T] += 0
        sum2[t] += 0
    # 记录添加完成之后的边数
    len_after = len(edges)
    # 搜索源汇点（未完成）并添加至edges
    # 附加源汇点,构造可行流求解图
    # 源点S和汇点T间建立一条虚拟边，虚拟变下界为0，上界为inf
    edges.append((T, S, INF))
    sum1[S] += 0
    sum2[T] += 0

    # 附加源汇SS,TT
    SS, TT = n + 1, n + 2
    for node in range(1, n + 1):
        v1 = 0 if node not in sum1 else sum1[node]
        v2 = 0 if node not in sum2 else sum2[node]
        if v1 > v2:
            edges.append((SS, node, v1 - v2))
        else:
            edges.append((node, TT, v2 - v1))

    Datasets = {
        'edges': edges,
        'SS': SS,
        'TT': TT,
        'max_node_num': n + 2,
        'max_edge_num': len(edges),
        'L': L,
        'edges_list': edges_list,
        'len_ori': len_ori,
        'len_after': len_after
    }

    return Datasets


def Dinic_2(edges_dict, low_l=0.5, n=6):
    """
    计算图的边有上下界容量问题的可行流分配

        采用Dinic 算法
        边可以有重边和自环边
        所有节点从1开始连续编号
        无固定边路定义该条边路下界默认为0.5, 上界为无穷
        
    param:
        edges_dict: (dict), 分支数据
        low_l: (number), 非固定风量分支下界
        n: (int), 节点风量误差控制, 相当于 10^-n
    
    return:
        返回初始化状态和分配的分支风量
    
    """

    Datasets = input(edges_dict, low_l)
    edges, source_node, end_node, max_node_num, max_edge_num = Datasets[
        'edges'], Datasets['SS'], Datasets['TT'], Datasets[
            'max_node_num'], Datasets['max_edge_num']

    # 初始化列表
    e = [-1] * (max_edge_num * 2 + 1)  # e[idx] 表示编号为idx残量图中边的终点
    f = [-1] * (max_edge_num * 2 + 1)  # f[idx] 表示编号为idx的残量图边的流量
    ne = [-1] * (max_edge_num * 2 + 1)  # ne[idx]表示根编号为idx的边同一个起点的下一条边的编号
    h = [-1] * (max_node_num + 1)  # h[a]表示以节点a为起点的所有边的链表头对应的边的编号
    dis = [-1] * (max_node_num + 1)  # dis[a] 表示点a到源点的距离，用于记录分层图信息
    # cur[a] 表示节点a在dfs搜索中第一次开始搜索的边的下标，数组cur记录点u之前循环到了哪一条边，以此来加速也称当前弧，用于优化dfs速
    cur = [-1] * (max_node_num + 1)
    # 用以增广流量
    orig_flow = [0] * (max_edge_num + 1)

    # 初始化列表的赋值
    idx = 0
    for _, (a, b, w) in enumerate(edges):
        e[idx], f[idx], ne[idx], h[a] = b, w, h[a], idx
        idx += 1
        e[idx], f[idx], ne[idx], h[b] = a, 0, h[b], idx
        idx += 1

    # bfs搜索有没有增广路
    def bfs() -> bool:
        for i in range(max_node_num + 1):
            # 表示点i到源点的距离，用于记录分层图信息
            dis[i] = -1
        # 源点处距离定义为0
        dis[source_node] = 0
        # 定义队列，寻找增广路径
        que = deque()
        que.append(source_node)
        # 记录当前弧
        cur[source_node] = h[source_node]
        # 循环队列，若有增广路则返回TRUE
        while len(que) > 0:
            cur_node = que.popleft()
            idx = h[cur_node]
            while idx != -1:
                next_node = e[idx]
                if dis[next_node] == -1 and f[idx] > 0:
                    dis[next_node] = dis[cur_node] + 1
                    cur[next_node] = h[next_node]
                    if next_node == end_node:
                        return True
                    que.append(next_node)
                idx = ne[idx]
        return False

    # dfs查找增广路
    def dfs(node, limit) -> int:
        """DFS查找增广路, 返回当前残量图上node节点能流入汇点的最大流量"""
        if node == end_node:
            return limit
        flow = 0
        # 从节点的当前弧开始搜索下一个点
        idx = cur[node]
        # 弧优化
        while idx != -1 and flow < limit:
            # 当前弧优化，记录每一个节点最后走的一条边
            # 另外一条路径到同一个点时候没必要重复搜索，已经不会再提供流量贡献的邻接点
            cur[node] = idx
            next_node = e[idx]
            if dis[next_node] == dis[node] + 1 and f[idx] > 0:
                t = dfs(next_node, min(f[idx], limit - flow))
                if t == 0:
                    # 已经无法提供流量的点不再参与搜索
                    dis[next_node] = -1
                # 更新残量图边的流量
                f[idx], f[idx ^ 1], flow = f[idx] - t, f[idx ^ 1] + t, flow + t
                # 更新原图边的流量
                if edges[idx >> 1][0] == node:
                    orig_flow[idx >> 1] += t
                else:
                    orig_flow[idx >> 1] -= t
            idx = ne[idx]
        return flow

    # 计算最大流量
    max_flow = 0
    while bfs():
        # 只要还有增广路，就dfs把增广路都找到，把增广路上的流量加到可行流上
        max_flow += dfs(source_node, INF)
    # 若无下界，以元组形式输出（最大流，[[起始点，终点，流量，权重]）
    # 若有下界，（最大流，[[起始点，终点，实际流量-下界，上界-下界]）
    ans = (max_flow, [[edges[i][0], edges[i][1], orig_flow[i], edges[i][2]]
                      for i in range(len(edges))])

    # 计算最大流，验证满流, 有负风量不能使用
    flag = True
    for a, b, f, w in ans[1]:
        if a == Datasets['SS'] and abs(f - w) > (1 / 10**n):
            # print('abs=======',abs(f - w))
            flag = False
            break
    # 计算构图
    # 输入数据
    if is_l(Datasets['L']):
        output = output_is_l(Datasets['L'], ans, Datasets['edges_list'])
        # 整理输出
        del output['edges'][Datasets['len_ori']:Datasets['len_after']]
    return flag, output["edges"]


def distribution(edges, minQ=0.5, n=6):
    """
    风量初始化风分配
    param:
        edges: (dict), 分支拓扑数据
        fixedQs: (dict), 固定风量数据
        minQ: (number), 非固定风量分支下界
        n: (int), 节点风量误差控制, 相当于 10^-n
    return:
        返回初始化状态和分配的分支风量"""

    # 最大流
    status, flow_distribution = Dinic_2(edges, low_l=minQ, n=n)

    # 分解拓扑与风量
    for obj in edges:
        del obj['fixedQ']

    return status, flow_distribution
