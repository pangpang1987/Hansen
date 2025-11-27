from math import inf
import numpy as np


def find_all_nodes(edges):
    '''找到图中所有的节点'''
    nodes = list()
    for i in edges:
        if i['s'] not in nodes:
            nodes.append(i['s'])
        if i['t'] not in nodes:
            nodes.append(i['t'])
    return nodes


def filter_node(edges):
    '''
        筛选出所有节点（不包括源汇点）
        返回：节点列表
    '''

    # 找到每条边的源点
    s_nodes = list()  # 源点
    t_nodes = list()  # 汇点
    for edge in edges:
        if edge['s'] not in s_nodes:
            s_nodes.append(edge['s'])
        if edge['t'] not in t_nodes:
            t_nodes.append(edge['t'])

    # 记录源汇点
    S_nodes = []
    T_nodes = []

    # 筛选只有入度和出度的汇点
    for i in s_nodes:
        if i not in t_nodes:
            S_nodes.append(i)
    for i in t_nodes:
        if i not in s_nodes:
            T_nodes.append(i)

    # 筛除源汇点
    num_m_nodes = list(set(s_nodes + t_nodes) - set(S_nodes + T_nodes))

    # 不包括源汇点的所有节点
    return num_m_nodes


def v_edges(edges):
    '''
    输出邻接矩阵edges列；权阵的建立以及改正数的顺序的参照
    '''
    return {edge['id']: 0 for edge in edges}


def bulid_P(edges):
    '''
    创建权阵P
    '''
    p = []
    for edge in edges:
        if edge['Q'] == 0:
            p.append(9999)
        else:
            p.append(1 / edge['Q'])
    return np.diag(p)


def bulid_maxtrix(edges):
    """
    创建系数矩阵A和W，输出风量flowQ
    """
    vertexs = filter_node(edges)

    A = []
    W = []
    flowQ = []

    for v in vertexs:
        # 节点邻接矩阵
        a = []
        # 闭合差
        w = []
        for edge in edges:
            if len(flowQ) != len(edges):
                flowQ.append(edge['Q'])
            if edge['s'] == v:
                a.append(1)
                w.append(edge['Q'])
            elif edge['t'] == v:
                a.append(-1)
                w.append(-edge['Q'])
            elif edge['s'] != v and edge['t'] != v:
                a.append(0)
                w.append(0)

        A.append(a)
        W.append(w)

    flowQ = np.array(flowQ)
    diffs = []
    for i in W:
        diff = []
        close_error = np.sum(i)
        diff.append(close_error)
        diffs.append(diff)
    return np.vstack(A), np.vstack(diffs), flowQ.reshape(-1, 1)


def adjustment(edges):
    """
    平差计算，输出中误差
    """
    # 处理网络拓扑和风量
    m = len(filter_node(edges))  # 节点数
    p_edges_keys = [edge["id"] for edge in edges]  # 权阵建立所用的边顺序

    # 矩阵处理
    P = bulid_P(edges)  # 权值
    A, W, flowQ = bulid_maxtrix(edges)  # 关联矩阵，闭合差矩阵，风量矩阵
    Q = np.linalg.inv(P)  # 协因数阵
    K = np.linalg.inv(A @ Q @ (A.T)) @ (-W)  # 联系数向量
    V = np.linalg.inv(P) @ A.T @ K  # 改正数
    # RMSE = np.sqrt((V.T @ P @ V) / m)  # 中误差
    # print(RMSE)

    # 修正后的风量
    adjutmentQ = V + flowQ
    edgeQ = []
    for i in adjutmentQ:
        edgeQ.append(round(i[0], 5))

    # 整理数据
    rough_output = dict(zip(p_edges_keys, edgeQ))  # 原始数据

    return rough_output
