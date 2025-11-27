#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	jGraph     通用，与深度优先不同，深度优先是专用的
#
###########################################################################################################################

# from    .i_graph_algorithm      import  context_i_graph_algorithm

# # import  sys
# # sys.setrecursionlimit(1000000000)        # 递归深度

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   节点类
#
class Node :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
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

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   节点度
    def get_dgree(self, mode='out') -> int :      # out/in/all
        def _out_()     :   return len(self.outEs)
        def _in_()      :   return len(self.inEs)
        def _all_()     :   return len(self.outEs) + len(self.inEs)
        func = {
            'out'       :   _out_,
            'in'        :   _in_,
            'all'       :   _all_
        }
        return func[mode]()
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   深度优先遍历搜索分支类
#        
class Edge :
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
#   
#
class GraphJ :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self, edges) :
        # 1. 创建节点、分支类对象
        self.edges = edges
        self.es         =   []          # 分支类对象集合
        self.vs         =   []          # 节点类对象集合
        vids            =   []          # 节点id集合
        self.mapv       =   {}          # 节点id映射类对象
        self.mape       =   {}
        self.me = {}

        for e in edges :
            self.me[e['id']] = e
            s,t = e['s'], e['t']
            if s not in vids :
                vids.append(s)
                sobj = Node(s)
                self.mapv[s] = sobj
                self.vs.append(sobj)
            if t not in vids :
                vids.append(t)
                tobj = Node(t)
                self.mapv[t] = tobj
                self.vs.append(tobj)
            if not e.get('weight', None) : weight = 1       # 默认1
            else : weight=e['weight']
            eobj = Edge(e['id'], s=self.mapv[s], t=self.mapv[t],weight=weight)
            self.es.append(eobj)
            self.mape[e['id']] = eobj
            self.mapv[s].outEs.append(eobj)
            self.mapv[t].inEs.append(eobj)

        # 5. 源汇节点、分支
        self.sourceVIds   = [v.id for v in self.vs if len(v.inEs)==0 and len(v.outEs)==1]        # id
        self.sinkVIds     = [v.id for v in self.vs if len(v.outEs)==0 and len(v.inEs)==1]        # id
        self.sourceEIds   = [e.id for e in self.es if len(e.s.inEs)==0 and len(e.s.outEs)==1]    # id
        self.sinkEIds     = [e.id for e in self.es if len(e.t.outEs)==0 and len(e.t.inEs)==1]    # id

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   搜索、避让始末点
    def get_start_target_nodes_and_avoid_edges(self, startNodes=None, targetNodes=None, avoidEdges=None) :
        # 1. 搜索始末点 
        if   startNodes         is  None    :   startNodes  =   self.sourceVIds       # 默认全部源点
        elif type(startNodes)   ==  str     :   startNodes  =   [startNodes]            # str型给参
        elif type(startNodes)   ==  list    :   startNodes  =   startNodes

        if   targetNodes        is  None    :   targetNodes =   self.sinkVIds       # 默认全部源点
        elif type(targetNodes)  ==  str     :   targetNodes =   [targetNodes]            # str型给参
        elif type(targetNodes)  ==  list    :   targetNodes =   targetNodes


        # 2. 避让分支id
        if   avoidEdges         is  None    :   avoidEdges  =   []
        elif type(avoidEdges)   ==  str     :   avoidEdges  =   [avoidEdges]
        elif type(avoidEdges)   ==  list    :   avoidEdges  =   avoidEdges

        return startNodes, targetNodes, avoidEdges
    

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   独立回路
    #   说明:
    #   （1）可以是非连通图
    #   （2）可以有并联分支
    #   （3）可以有单向回路
    #   （4）分支权重值必须 >= 0
    #   （5）可以无权重
    def get_circuits(self, weights=None) :
        with context_i_graph_algorithm(self.edges) as iGraph :
            minTree, coTree, circuits = iGraph.get_mintree_cotree_circuits(weights=weights)

        cs_ = list()
        for circuit in circuits :
            v0 = self.mape[circuit[0]].s.id
            c_ = list()
            for eid in circuit :
                eobj = self.mape[eid]
                if eobj.s.id == v0 : 
                    d = 1
                    v0 = eobj.t.id
                else :
                    d = -1
                    v0 = eobj.s.id
                c_.append({'eid':eid,'d':d})
            cs_.append(c_)
        return cs_, circuits, minTree, coTree

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索遍历着色
    def depth_first_search_traverses_coloring(
        self, 
        startNodes  =   None,       # None/str/list, 搜索起始点
        targetNodes =   None,       # None/str/list, 搜索目标点
        avoidEdges  =   None        # None/str/list, 避让分支
    ) :

        def _func_node_(vid, successEs, targetNodes, avoidEdges, track, snake) :

            v = self.mapv[vid]                        # 当前节点类对象        
            track.extend(['+',vid])              # 记录节点轨迹

            # 2. 当前节点为目标节点目标节点
            if vid in targetNodes :                  # 节点为目标节点
                v.success       =   True                # 搜索成功
                track.extend(['-',vid])          # 退栈  如果目标点不是汇点，能否出现目标点单向回路？？？
                return

            # 3. 单向回路回退   ？？？？？？   注意连续回退情况，暂未完成，目前仅限于无单向回路
            if vid in snake :           # 蛇头咬蛇身
                track.extend(['-',vid])      # 回退的是刚加入的，蛇身的未加入
                return
            else :
                snake.append(vid)    # 当前节点加入蛇头通路（无此步，前方单向回路不能避免）不能再判断单向回路之前

            # 4. 着色节点回退
            if v.color is True:
                track.extend(['-',vid])
                return

            # 6. 节点无出边回退
            if not len(self.mapv[vid].outEs) :
                track.extend(['-',vid])
                v.success = False
                return

            #7. 节点出边循环体
            for e in self.mapv[vid].outEs :           # 节点出边循环
                track.extend(['+',e.id])         # 记录轨迹

                if e.id in avoidEdges :            #当前出边为避让边
                    e.success=False  
                    continue

                t = e.t                                 # 当前出边末节点

                # 分支深度搜索
                _func_node_(t.id, successEs, targetNodes, avoidEdges, track, snake)
                e.success = t.success
                track.extend(['-',e.id])

            # 当前节点是否能到目标点
            for e in v.outEs : 
                if e.success is True : 
                    successEs.append(e.id)
                    v.success = True

            v.color = True

            snake.pop()
            track.extend(['-',vid])

        # 节点、分支颜色初始化，否则连续两次搜索出错
        for eobj in self.es : 
            eobj.color = None
            eobj.success = None

        for vobj in self.vs : 
            vobj.color = None
            vobj.success = None

        startNodes, targetNodes, avoidEdges = self.get_start_target_nodes_and_avoid_edges(
            startNodes=startNodes, 
            targetNodes=targetNodes, 
            avoidEdges=avoidEdges
        )

        successEs = []
        track = []      # 搜索轨迹
        snake = []      # 环形蛇
        for vid in startNodes :
            _func_node_(vid, successEs, targetNodes, avoidEdges, track, snake)

        return successEs, track


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
# #
# #   JGraph上下文
# #
# @contextmanager
# def context_jian_graph(edges):
#     jGraph = JianGraph(edges)
#     yield jGraph
# #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€



