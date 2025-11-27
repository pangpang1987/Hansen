#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   深度优先搜索法求全部通路模块
#                                                           #
######################################################################################

import  sys
from    jl.common.dataProcessing.globalVariable import *

sys.setrecursionlimit(100000000)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#   深度优先正向搜索全部通路函数
#
def GetPaths(top,startNodes=None,targetNodes=None,giveways=None) :
    """深度优先搜索全部通路函数
    :top----分支拓扑词典列表（仅需拓扑关系）
    :startNodes----搜索起始点id列表。startNodes=None,所有源点为搜索起始点
    :targetNodes----搜索目标点id列表。targetNodes=None,所有汇点为搜索目标点
    :giveway----避让分支id列表
    :paths----返回值,分支id集合列表
    """
    # print("top==========",top)
    # for e in top :
    #     print(e)
    #1: 设置起始节点、目标节点
    if not startNodes or not targetNodes :
        v_in = {}
        v_out = {}
        vs = set()
        for e in top :
            s,t = e[S], e[T]
            if s not in vs :
                v_in[s] = 0
                v_out[s] = 0
            if t not in vs :
                v_in[t] = 0
                v_out[t] = 0
            vs.update([s,t])
            v_out[s] = v_out[s]+1
            v_in[t] += v_in[t]+1
        if not startNodes :
            startNodes = []
            for vid,e_in in v_in.items() :
                if not e_in :
                    startNodes.append(vid)
        if not targetNodes :
            targetNodes = []
            for vid,e_out in v_out.items() :
                if not e_out :
                    targetNodes.append(vid)

    #2: 设置避让分支
    if not giveways : giveways = []

    #3: 创建节点数据类对象
    emap    = {}        # id-edge 映射
    vmap    = {}        # vid-node 映射
    vs      = []        # 节点数据类对象列表    
    for e in top :
        eid, s, t = e[ID], e[S], e[T]
        emap[eid] = e
        if not vmap.get(s,None) :
            v = NodeData(s)
            vmap[s] = v
            vs.append(v)
        if not vmap.get(t,None) :
            v = NodeData(t)
            vmap[t] = v
            vs.append(v)
        vmap[s].out.append(eid)

    for vid in targetNodes :
        vmap[vid].status = 1

    # print(targetNodes)
    # print(startNodes)

    paths = []
    snake = []
    for vid in startNodes :
        ps,bool = _DepthFirstSearchdfs(
                    vid,targetNodes,vmap,emap,giveways,snake)
        paths.extend(ps)
    
    return paths
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   深度优先搜索节点数据类
#
class NodeData :
    def __init__(self,vid) :
        self.vid    = vid       # 节点 id
        self.status = 0         # 节点搜索状态
        self.paths  = []        # 节点到目标点通路
        self.out    = []        # 节点出边
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   深度优先搜索
#
def _DepthFirstSearchdfs(vid,targetNodes,vmap,emap,giveways,snake) :
    #1: 解析当前节点
    node = vmap[vid]

    #2: 单向回路判断
    if vid in snake : return [],False

    #3: 节点搜索状态
    if node.status == 1 : return node.paths,True
    
    #4: 标记当前节点状态为完成
    node.status = 1

    #5: 当前节点记为蛇身
    snake.append(vid)

    #6: 节点无出边
    if not len(node.out) : return [],True

    #7: 节点出边循环体
    for eid in node.out :
        e = emap[eid]
        
        #当前出边为避让边
        if eid in giveways : continue

        # 当前出边末节点
        target = e[T]

        #出边末节点为目标节点
        if target in targetNodes :      # 出边为目标节点
            path = [eid]
            node.paths.append(path)
        #末节点非目标，深度搜索
        else :
            paths,bool = _DepthFirstSearchdfs(
                target,targetNodes,vmap,emap,giveways,snake)

            #有通路
            if len(paths) > 0 :
                for path in  paths:
                    node.paths.append([eid]+path)
            #无通路
            else :
                #汇点
                if bool :
                    giveways.append(eid)
                #单向
                # else :
                #     node.status = 0

    #蛇头退栈
    snake.pop()
    if not len(node.paths) and not bool :
        node.status = 0

    return node.paths,True
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
