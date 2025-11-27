#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	JianGraph类
#
###########################################################################################################################

import  sys
sys.setrecursionlimit(1000000000)        # 递归深度

from    .decorator  import  (
        decorator_node_target,          # 目标节点装饰器
        decorator_node_loop,            #
        decorator_node_color,
        decorator_node_no_out_edge,     # 节点无出边装饰器
        decorator_node_end,             # 当前节点搜索结束装饰器

        decorator_edge_giveway,         # 避让装饰器
        decorator_edge_search           # 当前分支搜索结束装饰器
)

from    .max_path     import  (
        get_all_max_paths       as  _get_all_max_paths_,
        get_max_path            as  _get_max_path_

)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   网络类
#
class JianGraph() :
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self, edges=[]) :
        
        # 1. 拓扑数据
        self.edges      =   edges

        # 2. 节点集合
        vs = []
        for e in edges :
            s, t = e['s'], e['t']
            vs.extend([s,t])
        self.nodes = list(set(vs))

        # 3. 分支id映射
        self.mape = dict()
        for e in self.edges :
            self.mape[e['id']] = e

        # 4. 节点入边、出边
        self.nodeInEdges            =   {**{v:[] for v in self.nodes}}
        self.nodeOutEdges           =   {**{v:[] for v in self.nodes}} 

        for e in self.edges :
            self.nodeInEdges[e['t']].append(e)
            self.nodeOutEdges[e['s']].append(e)

        # 5. 源汇节点、分支
        self.sourceNodes = [vid for vid, es in self.nodeInEdges.items() if len(es)==0]
        self.sinkNodes = [vid for vid, es in self.nodeOutEdges.items() if len(es)==0]
        self.sourceEdges = [e['id'] for e in self.edges if e['s'] in self.sourceNodes]
        self.sinkEdges = [e['id'] for e in self.edges if e['t'] in self.sinkNodes]

         
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


    def get_max_path(self, starts=None, targets=None, weight=None) -> list :
        return _get_max_path_(self, starts=starts, targets=targets, weight=weight)


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   全部源汇点对最长路
    def get_all_max_paths(self, starts=None, targets=None, weight=None) -> list :
        return _get_all_max_paths_(self, starts=starts, targets=targets, weight=weight)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   通路
    def get_paths(self, starts=None, targets=None, mode='number') -> list :    # path通路，number复杂度，all全部
        func    =   'ALL_PATHS'
        vpaths = {}                         # 节点正向通路
        vpathNumber = {}                    # 节点通路数
        epaths = {}                         # 分支正向通路
        epathNumber = {}                    # 分支通路数

        for e in self.edges :   
            epaths[e['id']], epathNumber[e['id']] = [], 0
        for vid in self.nodes :
            vpaths[vid], vpathNumber[vid] = [], 0

        data = dict()
        data['vpaths']          =   vpaths
        data['epaths']          =   epaths
        data['epathNumber']     =   epathNumber
        data['vpathNumber']     =   vpathNumber
        data['mode']            =   mode

        self.depth_first_traversal_search(
            starts=starts,
            targets=targets,
            data    =   data,
            func    =   func
        )

        paths = []
        sum = 0
        if not starts :                                     # 默认全部源点
            starts = []
            for vid, es in self.nodeInEdges.items() :
                if len(es) == 0 :
                    starts.append(vid)
        for vid in starts :
            paths.append(data['vpaths'][vid])
            sum += data['vpathNumber'][vid]

        if      mode    ==  'all'       :   return  paths, sum
        elif    mode    ==  'number'    :   return  None,sum



    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   遍历着色（截流过滤等）
    def get_eids_color(self, starts=None, targets=None, giveways=None) -> set :     # [{eid : []}, ...]
        func    =   'TRAVERSAL_COLOR'
        ecolor = {}                                             # 分支关联汇点
        for e in self.edges     :   ecolor[e['id']] = False
        vcolor = {}                                             # 节点关联汇点
        for vid in self.nodes   :   vcolor[vid] = False
        data = dict()
        data['ecolor'] = ecolor
        data['vcolor'] = vcolor
        self.depth_first_traversal_search(
            starts=starts,
            targets=targets,
            data    =   data,
            func    =   func,
            giveways=giveways
        )
        color = list()
        colorless = list()
        for eid, c in data['ecolor'].items() :
            if c : color.append(eid)
            else : colorless.append(eid)
        return  color, colorless
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   汇点子网
    def set_edge_associated_sink_node(self) -> dict :     # [{eid : []}, ...]
        func    =   'SINK_NODE_SUB_G'
        esinks = {}                                             # 分支关联汇点
        for e in self.edges     :   esinks[e['id']] = []
        vsinks = {}                                             # 节点关联汇点
        for vid in self.nodes   :   vsinks[vid] = []
        data = dict()
        data['esinks'] = esinks
        data['vsinks'] = vsinks
        self.depth_first_traversal_search(
            data    =   data,
            func    =   func
        )
        return  data['esinks']
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先遍历搜索
    def depth_first_traversal_search(
            self, 
            starts      =   None, 
            targets     =   None, 
            giveways    =   None, 
            data        =   None,
            mode        =   'out',
            func        =   None
    ) :
        
        # 1. 搜索起始点
        if not starts :                                     # 默认全部源点
            starts = []
            for vid, es in self.nodeInEdges.items() :
                if len(es) == 0 :
                    starts.append(vid)
        elif type(starts) == str :                          # str型给参
            starts = [starts]

        if not targets :                                    # 默认全部汇点
            targets = []
            for vid, es in self.nodeOutEdges.items() :
                if len(es) == 0 :
                    targets.append(vid)
        elif type(targets) == str :
            targets = [targets]
        
        # 2. 定义
        path = []
        snake = []
        nodeColor=[]
        if not giveways : giveways=[]

            
        # vpaths = {}     # 节点通路集
        # for vid in self.nodes :
        #     vpaths[vid] = []
        # data['vpaths'] = vpaths


        for s in starts :
            self.depth_first_traversal_search_node_func(            # 节点函数
                s, 
                targets,
                path, 
                giveways, 
                snake, 
                nodeColor, 
                data=data,                
                mode=mode,                 # 正向
                funcname=func
            )
    #   返回值函数  通路极限
    #   找通路，污染范围、回风井子网（回风井公共分支）
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先遍历搜索节点函数
    #   仅搜索一次，    注意目标点不要着色，原因是目标点可有多个入边，将目标点排在首位也可解决着色带来的丢通路等问题
    def depth_first_traversal_search_node_func(
            self, 
            vid, 
            targetNodes, 
            path, 
            giveways, 
            snake, 
            nodeColor, 
            data=None,
            mode='out', 
            funcname=None
    ) :
        # print("vid===================",vid)

        # 1. 定义深度优先遍历搜索内部功能函数
        # 1.1 节点类函数
        # 1.1.1 目标节点
        @decorator_node_target
        def node_target(funcname, *args, **kwargs) -> dict : pass
        # 1.1.2 单向回路节点
        @decorator_node_loop
        def node_loop(funcname, *args, **kwargs) -> dict : pass
        # 1.1.3 着色节点
        @decorator_node_color
        def node_color(funcname, *args, **kwargs) -> dict : pass
        # 1.1.4 无出边节点
        @decorator_node_no_out_edge
        def node_no_out_edge(funcname, *args, **kwargs) -> dict : pass
        # 1.1.5 节点结束回退
        @decorator_node_end
        def node_end(funcname, *args, **kwargs) -> dict : pass
        # 1.2 分支类函数
        # 1.2.1 避让
        @decorator_edge_giveway
        def edge_giveway(funcname, *args, **kwargs) -> dict : pass
        # 1.2 分支深度优先遍历搜索返回值处置函数
        @decorator_edge_search
        def edge_search(funcname, *args, **kwargs) -> dict : pass


        # 2. 深度优先遍历搜索节点函数

        # 2.1 目标节点回退（注意目标节点在前，着色节点在后，否则，丢通路）
        if vid in targetNodes :                                                    # 节点为目标节点
            node_target(funcname,vid=vid,path=path,snake=snake,data=data)           # 调用目标节点函数
            return

        # 2.2 单向回路回退
        if vid in snake :           # 蛇头咬蛇身
            node_loop(funcname,vid=vid,path=path,snake=snake,data=data)
            return

        # 2.3 着色节点回退
        if vid in nodeColor :       # 节点已着色
            node_color(funcname,vid=vid,path=path,snake=snake,data=data)
            return

        # 2.4 当前节点加入搜索通路（无此步，前方单向回路不能避免）
        snake.append(vid)

        # 2.5 节点无出边回退
        if not len(self.nodeOutEdges) :
            node_no_out_edge(funcname,vid=vid,path=path,snake=snake,data=data)
            return

        #2.6 节点出边循环体
        for e in self.nodeOutEdges[vid] :
            # print("eid===============",eid)

            # e = self.mape[eid]           # 当前分支  dict

            if e['id'] in giveways :            #当前出边为避让边
                edge_giveway(funcname,vid=vid,path=path,snake=snake,data=data)       # 一般不操作      
                continue

            t = e['t']                      # 当前出边末节点

            path.append(e['id'])                # 搜索通路延长

            self.depth_first_traversal_search_node_func(        # 当前分支末节点深度搜索
                vid=t, 
                targetNodes=targetNodes, 
                path=path, 
                giveways=giveways, 
                snake=snake, 
                nodeColor=nodeColor, 
                data=data,
                mode=mode, 
                funcname=funcname
            )

            edge_search(                           # 当前分支深度优先遍历搜索返回值处置函数
                 funcname,
                 edge=e,
                 vid=vid,
                 t=t,
                 path=path,
                 data=data
            )
                
            giveways.append(e['id'])

            path.pop()              # 通路退栈

            # print("退出通路------",e)

        nodeColor.append(vid)       # 节点着色
        snake.pop()                 #蛇头退栈

        node_end(
            funcname, 
            vid=vid,
            nodeOutEdges=self.nodeOutEdges[vid],
            data=data
        )
        return
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
