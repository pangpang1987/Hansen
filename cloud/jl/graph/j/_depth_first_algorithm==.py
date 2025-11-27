#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	depth_first_algorithm 深度优先正向搜索法模块    是一个独立模块
#
###########################################################################################################################

import  sys
sys.setrecursionlimit(1000000000)        # 递归深度

from    contextlib                      import  contextmanager

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   深度优先遍历搜索节点类
#
class Node1 :
    def __init__(self, id) -> None:
        self.id = id
        self.color = None
        self.outEs = []
        self.inEs = []
        # self.paths = {}             # 节点到各目标点所有通路
        self.data   =   {}
        # self.maxHs  = {}            # 节点到各目标点权重之和sum_weight
        # self.maxPaths = {}           # 节点到各目标点最长通路
        self.maxPath = []           # 全风网最常路
        self.maxH = 0            # 全风网节点通路最大值
        # self.minH   =   None
        # self.minPath = []
        self.pathCount = 0           # 通路数

        self.success    =   None    # True,False 搜索成功，出边有一个成功，节点就成功
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   深度优先遍历搜索分支类
#        
class Edge1 :
    def __init__(self, id, s=None, t=None, weight=None) -> None:
        self.id = id
        self.s = s      # class obj
        self.t = t      # class obj
        self.color = None
        self.weight = weight
        self.data   =   {}
        # self.paths = {}             # 分支到各目标点所有通路
        # self.maxHs  = {}            # 分支到各目标点权重之和sum_weight
        # self.maxPaths = {}           # 分支到各目标点最长通路
        self.maxPath = []           # 全风网最长路
        self.maxH = 0            # 全风网分支通路最大值
        self.pathCount = 0
        self.success    =   None    # True,False 搜索成功，末节点成功分支就成功
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   DepthFirstTraversalSearchGraph 深度优先遍历搜索图类
#
#   
class DepthFirstTraversalSearchGraph :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self,                                  # 构造函数 
        edges,                                          # 有向图数据
        source                          =   None, 
        to                              =   None, 
        giveways                        =   None, 
        data                            =   {},
        mode                            =   'out'
    ) :
        # 1. 创建节点、分支类对象
        self.es         =   []          # 分支类对象集合
        self.vs         =   []          # 节点类对象集合
        vids            =   []          # 节点id集合
        self.mapv       =   {}          # 节点id映射类对象
        self.mape       =   {}

        for e in edges :
            s,t = e['s'], e['t']
            if s not in vids :
                vids.append(s)
                sobj = Node1(s)
                self.mapv[s] = sobj
                self.vs.append(sobj)
            if t not in vids :
                vids.append(t)
                tobj = Node1(t)
                self.mapv[t] = tobj
                self.vs.append(tobj)
            if not e.get('weight', None) : weight = 1       # 默认1
            else : weight=e['weight']
            eobj = Edge1(e['id'], s=self.mapv[s], t=self.mapv[t],weight=weight)
            self.es.append(eobj)
            self.mape[e['id']] = eobj
            self.mapv[s].outEs.append(eobj)
            self.mapv[t].inEs.append(eobj)

        # 5. 源汇节点、分支
        self.sourceVIds   = [v.id for v in self.vs if len(v.inEs)==0 and len(v.outEs)==1]        # id
        self.sinkVIds     = [v.id for v in self.vs if len(v.outEs)==0 and len(v.inEs)==1]        # id
        self.sourceEIds   = [e.id for e in self.es if len(e.s.inEs)==0 and len(e.s.outEs)==1]    # id
        self.sinkEIds     = [e.id for e in self.es if len(e.t.outEs)==0 and len(e.t.inEs)==1]    # id


        # 1. 搜索始末点 
        if   not     source      :  self.starts  =   self.sourceVIds       # 默认全部源点
        elif type(source) == str :  self.starts  =   [source]            # str型给参
        elif type(source) == list : self.starts = source

            
        if   not     to          :  self.targets      =   self.sinkVIds                               # 默认全部汇点
        elif type(to) == str     :  self.targets      =   [to]
        elif type(to) == list : self.targets = to
            
        
        # 2. 避让分支id
        if not giveways : self.giveways=[]      # id
        else : self.giveways = giveways

        if not data : self.data = {}
        else : self.data=data

        self.track = []             # 搜索轨迹
        self.snake = []     # id
        # self.path = []      # id
        # self.paths = []     # id
        # self.mode=mode
        self.successEs = []

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   DepthFirstTraversalSearchGraph上下文
#
@contextmanager
def context_depth_first_traversal_search_graph(
        edges,
        source                          =   None, 
        to                              =   None, 
        giveways                        =   None 
    ):

    g = DepthFirstTraversalSearchGraph(
        edges,                                          # 有向图数据
        source      =   source,                     # 与igraph统一，内部用starts, source是网络源点，不是搜索源点
        to          =   to,                         # 内部用targets
        giveways    =   giveways
        # mode='out'
    )
    yield g
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   DepthFirstAlgorithm 深度优先正向遍历搜索法类
#
class DepthFirstAlgorithm() :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(
        self,
        g,
        data                            =   None,
        func_to_node                    =   None,       # 目标节点函数
        func_node_no_have_out_edge      =   None,       # 节点没有出边函数
        func_node_search_finish_back    =   None,       # 节点搜索结束回溯函数
        func_edge_search_finish         =   None,       # 分支搜索结束函数
        func_snak_head_touches_tail     =   None,       # 单向回路函数
        func_giveways                   =   None,       # 避让函数
        func_touches_coloured_node      =   None,       # 碰到着色节点
        func_touches_coloured_edge      =   None        # 碰到着色分支    
    
    ) -> None:
        
        self.g                              =   g                                   # 搜索图
        self.data                           =   data                                # 外部数据

        self.func_to_node                   =   func_to_node
        self.func_node_no_have_out_edge     =   func_node_no_have_out_edge       # 节点没有出边函数
        self.func_node_search_finish_back    =   func_node_search_finish_back       # 节点搜索结束回溯函数
        self.func_edge_search_finish         =   func_edge_search_finish       # 分支搜索结束函数
        self.func_snak_head_touches_tail     =   func_snak_head_touches_tail       # 单向回路函数
        self.func_giveways                   =   func_giveways       # 避让函数
        self.func_touches_coloured_node      =   func_touches_coloured_node       # 碰到着色节点
        self.func_touches_coloured_edge      =   func_touches_coloured_edge       # 碰到着色分支        


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先遍历搜索开始函数
    def search_start(self) :
        for sid in self.g.starts : self.node_recursion_depth_first_traversal_search_func(sid)       # 起始节点循环

        # 整理返回数据  交由外部整理

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   节点递归深度优先遍历搜索函数    注意通路数采用逆推法，一遍搜尽
    def node_recursion_depth_first_traversal_search_func(self, vid) :

        # 1. 提取当前节点类对象、记录搜索轨迹

        v = self.g.mapv[vid]                        # 当前节点类对象        
        self.g.track.extend(['+',vid])              # 记录节点轨迹

        # 2. 当前节点为目标节点目标节点
        if vid in self.g.targets :                  # 节点为目标节点
            # v.maxH          =   float(0)            # 
            v.pathCount     =   1                   # 通路数
            v.success       =   True                # 搜索成功
            if self.func_to_node : self.func_to_node(self.g,vid)    # 外部函数
            self.g.track.extend(['-',vid])          # 退栈  如果目标点不是汇点，能否出现目标点单向回路？？？
            return

        # 3. 单向回路回退   ？？？？？？   注意连续回退情况，暂未完成，目前仅限于无单向回路
        #   考虑节点成功情形---------------------！！！！！！！！！！！！！！！！？？？？？？？？？？？
        if vid in self.g.snake :           # 蛇头咬蛇身
            if self.func_snak_head_touches_tail : self.func_snak_head_touches_tail(self.g, vid)
            self.g.track.extend(['-',vid])      # 回退的是刚加入的，蛇身的未加入
            return
        else :
            self.g.snake.append(vid)    # 当前节点加入蛇头通路（无此步，前方单向回路不能避免）不能再判断单向回路之前

        # 4. 着色节点回退
        if v.color is True:
            # print('-------------------着色回退')
            if self.func_touches_coloured_node : self.func_touches_coloured_node(self.g, vid)
            self.g.track.extend(['-',vid])
            return

        # 6. 节点无出边回退
        if not len(self.g.mapv[vid].outEs) :
            if self.func_node_no_have_out_edge : self.func_node_no_have_out_edge(self.g,vid)
            self.g.track.extend(['-',vid])
            v.success = False
            return

        #7. 节点出边循环体
        for e in self.g.mapv[vid].outEs :           # 节点出边循环
            self.g.track.extend(['+',e.id])         # 记录轨迹

            if e.id in self.g.giveways :            #当前出边为避让边
                e.success=False 
                if self.func_giveways   :   self.func_giveways(self.g, e)    
                continue

            t = e.t                                 # 当前出边末节点

            # 分支深度搜索
            self.node_recursion_depth_first_traversal_search_func(t.id)        # 递归深度优先搜索
            e.success = t.success

            e.pathCount = t.pathCount
            e.maxH = t.maxH + e.weight
            if t.pathCount > 0 :
                e.maxPath = [e.id] + t.maxPath
            if self.func_edge_search_finish : self.func_edge_search_finish(self.g, e)
            self.g.track.extend(['-',e.id])


        #  当前节点搜索完毕准备回溯
        if self.func_node_search_finish_back : self.func_node_search_finish_back(self.g, vid)   # 外部函数


        # 计算节点通路数、最大阻力、确当当前节点是否成功
        count = 0
        maxH = 0
        maxPath = [] 
        for e in v.outEs : 
            count += e.pathCount
            if e.success is True : 
                self.g.successEs.append(e.id)
                v.success = True
            # else :                    # 错误，有一个成功即成功，不成功误操作
            #     v.success = False
            if e.maxH > maxH :
                maxH = e.maxH
                maxPath = e.maxPath  

        v.pathCount = count
        v.maxH = maxH
        v.maxPath = maxPath
        v.color = True

        self.g.snake.pop()
        self.g.track.extend(['-',vid])
        return
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   DepthFirstAlgorithm上下文
#
@contextmanager
def context_depth_first_algorithm(
        g,
        data                            =   None,
        func_to_node                    =   None,       # 目标节点函数
        func_node_no_have_out_edge      =   None,       # 节点没有出边函数
        func_node_search_finish_back    =   None,       # 节点搜索结束回溯函数
        func_edge_search_finish         =   None,       # 分支搜索结束函数
        func_snak_head_touches_tail     =   None,       # 单向回路函数
        func_giveways                   =   None,       # 避让函数
        func_touches_coloured_node      =   None,       # 碰到着色节点
        func_touches_coloured_edge      =   None        # 碰到着色分支    
    ):
    algo = DepthFirstAlgorithm(
        g=g,
        data=data,      # 外部数据
        func_to_node                    =   func_to_node,       # 目标节点函数
        func_node_no_have_out_edge      =   func_node_no_have_out_edge,       # 节点没有出边函数
        func_node_search_finish_back    =   func_node_search_finish_back,       # 节点搜索结束回溯函数
        func_edge_search_finish         =   func_edge_search_finish,       # 分支搜索结束函数
        func_snak_head_touches_tail     =   func_snak_head_touches_tail,       # 单向回路函数
        func_giveways                   =   func_giveways,       # 避让函数
        func_touches_coloured_node      =   func_touches_coloured_node,       # 碰到着色节点
        func_touches_coloured_edge      =   func_touches_coloured_edge        # 碰到着色分支  
    )
    yield algo
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   深度优先遍历搜索确定通路数、通路最大长度、最大通路
#
def get_sumpath_maxh_maxpath_depth_first_traversal_search(
        edges, 
        source      =   None, 
        to          =   None, 
        giveways    =   []
    ) :

    with context_depth_first_traversal_search_graph(
        edges,
        source      =   source, 
        to          =   to, 
        giveways    =   giveways    
    )   as  g : pass

    with context_depth_first_algorithm(
        g,
        data={},
        func_to_node                    =   None,       # 目标节点函数
        func_node_no_have_out_edge      =   None,       # 节点没有出边函数
        func_node_search_finish_back    =   None,       # 节点搜索结束回溯函数
        func_edge_search_finish         =   None,       # 分支搜索结束函数
        func_snak_head_touches_tail     =   None,       # 单向回路函数
        func_giveways                   =   None,       # 避让函数
        func_touches_coloured_node      =   None,       # 碰到着色节点
        func_touches_coloured_edge      =   None        # 碰到着色分支  
    )   as algo :

        algo.search_start()             # 开始搜索

        # print(g.track)        # 搜索轨迹
        # for v in g.vs:print('v-id,count-maxH=====maxPath========',v.id,v.pathCount,v.maxH,v.maxPath)
        # for e in g.es:print('v-id,count-maxH=============',e.id,e.pathCount,e.maxH)
        # for vid in g.starts :print('----------------------',vid,g.mapv[vid].id)
        # for e in g.es :
        #     print(e.id, e.success)
        # for v in g.vs :
        #     print(v.id,v.success)
        count = 0
        maxH = 0
        maxPath = None
        for vid in g.starts :
            v = g.mapv[vid]
            count += v.pathCount
            if v.maxH > maxH :
                maxH = v.maxH
                maxPath = v.maxPath

        #     print('maxH=======',vid, g.mapv[vid].maxH,g.mapv[vid].maxPath)
        # print(count,maxH,maxPath)
        # print(g.successEs)

        return count, maxH, maxPath, g.successEs
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

 

