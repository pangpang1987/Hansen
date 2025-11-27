#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	GraphJ类模块
#
###########################################################################################################################

#   要点：该模块主要用于掘进巷删除和串联简化

from    dataclasses     import  (
        dataclass,                      # dataclass装饰器
        field                           # field函数
)
from    typing          import  List

import  sys
sys.setrecursionlimit(1000000000)        # 递归深度

from    contextlib import  contextmanager


from pprint import  pprint

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   节点类
#
@dataclass
class Node :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   
    id: str = field(
        default     =   None,
        compare     =   True,
    )

    ins: List = field(
        default_factory =   list,
        metadata    =   {
            'name'        :   '节点入边id列表'                                            
        }
    )
    outs: List = field(
        default_factory =   list
    )

    isok: bool = field(
        default     =   None,
        metadata    =   {
            'name'        :   '成功与否标识'               # True,False 搜索成功，出边有一个成功，节点就成功                              
        }
    )

    color: int = field(
        default=None,
        metadata={
            'name' : '着色标识'
        }
    )
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   节点度  用igraph
    def get_dgree(self, mode='out') -> int :      # out/in/all
        if      mode    ==  'out'   :   return  len(self.outs)
        elif    mode    ==  'in'    :   return  len(self.ins)
        elif    mode    ==  'all'   :   return  len(self.outs) + len(self.ins)
        return 
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   深度优先遍历搜索分支类
#
@dataclass
class Edge :

    id: str = field(
        default     =   None,
        compare     =   True,
    )

    s: str = field(
        default     =   None,
        compare     =   True,
    )

    t: str = field(
        default     =   None,
        compare     =   True,
    )

    color: int = field(
        default=None,
        metadata={
            'name' : '着色标识'
        }
    )

    isok: bool = field(
        default     =   None,
        metadata    =   {
            'name'        :   '成功与否标识'               # True,False 搜索成功，出边有一个成功，节点就成功                              
        }
    )

    weight: float = field(
        default=None
    )

        # self.weight = weight
        # self.data   =   {}
        # self.paths = {}             # 分支到各目标点所有通路
        # self.maxHs  = {}            # 分支到各目标点权重之和sum_weight
        # self.maxPaths = {}           # 分支到各目标点最长通路
        # self.maxPath = []           # 全风网最长路
        # self.maxH = 0            # 全风网分支通路最大值
        # self.pathCount = 0
        # self.ok    =   None    # True,False 搜索成功，末节点成功分支就成功
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   
#
class GraphJ :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self, edges=None) :
        # 1. 创建节点、分支类对象
        self.edges = edges
        self.es         =   []          # 分支类Edge对象集合
        self.vs         =   []          # 节点类Node对象集合

        self.mapv       =   {}          # 节点id映射类对象
        self.mape       =   {}
        self.me = {}

        vids            =   []          # 节点id集合

        for e in edges :
            # self.me[e['id']] = e
            eid, s, t = e['id'], e['s'], e['t']
            if s not in vids :
                vids.append(s)
                sobj = Node(id=s)
                self.mapv[s] = sobj
                self.vs.append(sobj)
            if t not in vids :
                vids.append(t)
                tobj = Node(id=t)
                self.mapv[t] = tobj
                self.vs.append(tobj)
            if not e.get('weight', None) : weight = 1       # 默认1
            else : weight=e['weight']
            eobj = Edge(id=eid, s=s, t=t, weight=weight)
            self.es.append(eobj)
            self.mape[eid] = eobj
            self.mapv[s].outs.append(eid)
            self.mapv[t].ins.append(eid)

        # 5. 源汇节点、分支
        self.sourceVIds   = [v.id for v in self.vs if len(v.ins)==0 and len(v.outs)==1]        # id
        self.sinkVIds     = [v.id for v in self.vs if len(v.outs)==0 and len(v.ins)==1]        # id
        self.sourceEIds   = [e.id for e in self.es if len(self.mapv[e.s].ins)==0 and len(self.mapv[e.s].outs)==1]    # id
        self.sinkEIds     = [e.id for e in self.es if len(self.mapv[e.t].outs)==0 and len(self.mapv[e.t].ins)==1]    # id

        # for e in self.es:
        #     print(e.id,e.s,e.t)
        # for v in self.vs:
        #     print(v.id,v.ins,v.outs)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   正向搜索串联边
    #   从eid开始正向搜索并返回串联边, eid入边部分缺失
    def find_edges_series_forward_eid(self, eid0) :
        es = []
        def _find_edge_(vid) :
            v = self.mapv[vid]
            if all([len(v.outs)==1, len(v.ins)==1]):
                eid = v.outs[0]
                es.append(eid)
                e = self.mape[eid]
                _find_edge_(e.t)
            else : return
        e = self.mape[eid0]
        _find_edge_(e.t)
        if len(es) == 0 : return []
        es.insert(0,eid0)
        return es
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   反向搜索串联边, eid出边部分缺失
    def find_edges_series_reverse_eid(self, eid0) :
        es = []
        def _find_edge_(vid) :
            v = self.mapv[vid]
            if all([len(v.ins)==1, len(v.outs)==1]):
                eid = v.ins[0]
                es.insert(0,eid)
                e = self.mape[eid]
                _find_edge_(e.s)
            else : return
        e = self.mape[eid0]
        _find_edge_(e.s)
        if len(es) == 0 : return []
        es.append(eid0)
        return es
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取eid串联边
    def get_edges_series_eid(self, eid0) -> list:
        es1 = self.find_edges_series_forward_eid(eid0)      # eid0正向串联
        es2 = self.find_edges_series_reverse_eid(eid0)      # eid0逆向串联

        if all([len(es1)>2, len(es2)>2])    :   return es2 + [eid for eid in es1 if eid!=eid0]  # 删除一个eid0
        elif len(es1) >= 2                  :   return es1
        elif len(es2) >= 2                  :   return es2
        else                                :   return []

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取vid串联边       vid端点可能存在多条
    def get_edges_series_vid(self, vid) -> list:
        # 1. 解析vid
        v = self.mapv[vid]

        # 2. vid单进单出构成串联
        if all([v.get_dgree('in')==1,v.get_dgree('out')==1]):
            return self.get_edges_series_eid(v.outs[0])
        
        # 3. vid多进多出
        seriess = []
        for eid in v.outs:
            series = self.get_edges_series_eid(eid)
            if len(series) > 0:
                seriess.append(series)
        for eid in v.ins:
            series = self.get_edges_series_eid(eid)
            if len(series) > 0:
                seriess.append(series)
        return seriess
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取全风网串联边
    def get_edges_series_all(self):
        # 1. 全部串联节点
        vs_series = [v for v in self.vs if all([v.get_dgree('in')==1,v.get_dgree('out')==1])]

        # 2. 定义列表容器
        colors = []              # 已着色串联节点
        seriess = []                # 串联集

        # 3. 遍历串联点求串联路
        for v in vs_series:
            if v.id in colors : continue
            e_in = v.ins[0]
            e_out = v.outs[0]
            series_in = self.find_edges_series_reverse_eid(e_in)
            if len(series_in) == 0  :   series_in = [e_in]
            series_out = self.find_edges_series_forward_eid(e_out)
            if len(series_out) == 0 : series_out = [e_out]
            series = series_in + series_out
            for eid in series:
                colors.append(self.mape[eid].s)
                colors.append(self.mape[eid].t)
            seriess.append(series)

        return seriess
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   串联简化
    def graph_series_simplification(self) -> list:
        seriess_eid = self.get_edges_series_all()                           # 所有串联[[eid]]
        filter_eids = [eid for series in seriess_eid for eid in series]     # 所有串联边id
        eid_map_series = {}
        series_map_eids = {}
        seriess_edge = []
        for i, series in enumerate(seriess_eid):
            # new_id = 'new_%d' % (i+1)
            edge = dict(zip(['id','s','t'],['series'+str(i+1), self.mape[series[0]].s,self.mape[series[-1]].t]))
            series_map_eids[edge['id']] = []
            for eid in series:
                series_map_eids[edge['id']].append(eid)
                eid_map_series[eid] = edge['id']
            seriess_edge.append(edge)
        
        return seriess_edge, eid_map_series, series_map_eids, filter_eids
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   搜索图中全部串联边
    def find_edges_series_all(self) :
        edges_series = []
        vdgree2 = []
        for v in self.vs :
            if len(v.ins)==1 and len(v.outs)==1 :
                v.color = None
                vdgree2.append(v)
        for v in vdgree2 :
            if v.color : continue
            print(v.id)
            es = self.find_edges_series_bidirectional(v.id)
            edges_series.append(es)
            vs = [self.mape[eid].t for eid in es]
            vs.pop()
            for v in vs : v.color = 1
        return edges_series
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
     
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索遍历着色
    def depth_first_search_traverses_coloring(
        self, 
        start  =   None,       # None/str/list, 搜索起始点, None全部源点
        to =   None,       # None/str/list, 搜索目标点, None全部汇点
        avoidEdges  =   None,       # None/str/list, 避让分支
        record      =   False       # 记录搜索轨迹
    ) :

        # 1. 定义搜索结果变量
        coloring    = []            # 着色边
        # track       = []            # 搜索轨迹
        snake       = []            # 环形蛇

        # 2. 定义节点搜索函数

        def _func_node_(vid, targetNodes, avoidEdges) :

            # 1. 解析当前节点类对象
            v = self.mapv[vid]                          # 当前节点类对象
            # if record : track.extend(['+',vid])         # 记录节点轨迹

            # 2. 当前节点为目标节点
            if vid in targetNodes :                     # 节点为目标节点
                v.isok       =   True                   # 搜索成功
                # if record : track.extend(['-',vid])     # 退栈  如果目标点不是汇点，能否出现目标点单向回路？？？
                return

            # 3. 单向回路回退   ？？？？？？   注意连续回退情况，暂未完成，目前仅限于无单向回路
            if vid in snake :                           # 蛇头咬蛇身
                # if record : track.extend(['-',vid])     # 回退的是刚加入的，蛇身的未加入
                return
            else :
                snake.append(vid)                        # 当前节点加入蛇头通路（无此步，前方单向回路不能避免）不能再判断单向回路之前

            # 4. 着色节点回退
            if v.color is True:
                # if record : track.extend(['-',vid])
                return

            # 5. 节点无出边回退
            if not len(self.mapv[vid].outs) :
                # if record : track.extend(['-',vid])
                v.isok = False
                return

            # 6. 节点出边循环体
            for eid in self.mapv[vid].outs :            # 节点出边循环
                # if record : track.extend(['+',eid])     # 记录轨迹

                if eid in avoidEdges :                  #当前出边为避让边
                    self.mape[eid].isok = False  
                    continue

                tid = self.mape[eid].t                  # 当前出边末节点

                # 分支深度搜索
                _func_node_(tid, targetNodes, avoidEdges)
                
                self.mape[eid].isok = self.mapv[tid].isok
                # if record : track.extend(['-',eid])

            # 当前节点是否能到目标点    有一条出边成功，则节点成功
            for eid in v.outs : 
                if self.mape[eid].isok is True : 
                    coloring.append(eid)
                    v.isok = True

            v.color = True

            snake.pop()
            # track.extend(['-',vid])

        # 主函数开始标识====================================================

        # 1. 节点、分支颜色初始化，否则连续两次搜索出错
        for eobj in self.es : 
            eobj.color = None
            eobj.isok = None          # 搜索成功表示

        for vobj in self.vs : 
            vobj.color = None
            vobj.isok = None

        if   isinstance(start, str)     :   start = [start]
        elif isinstance(start, list)    :   pass
        elif start is None              :   start = self.sourceVIds
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")

        if   isinstance(to, str)        :   to = [to]
        elif isinstance(to, list)       :   pass
        elif to is None                 :   to = self.sinkVIds
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")

        if   isinstance(avoidEdges, str)        :   avoidEdges = [avoidEdges]
        elif isinstance(avoidEdges, list)       :   pass
        elif avoidEdges is None                 :   pass
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")
            

        # 4. 开始搜索
        for vid in start :     # 遍历起始点
            _func_node_(vid, to, avoidEdges)

        # 5.返回
        return coloring
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   JGraph上下文
#
@contextmanager
def context_jian_graph(edges):
    jGraph = JianGraph(edges)
    yield jGraph
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€



