#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	图类模块
#
###########################################################################################################################



#   1. 仅设置通风网络常用的函数, 非常用函数直接调用GraphI, GraphJ

from    .graph_i            import  GraphI
from    .graph_j            import  GraphJ
from    .i.get_circuits     import  get_circuits


from pprint import  pprint

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   图类
#
class G :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   初始化函数
    def __init__(self, *args, edges=None, **kwargs):
        self.edges = edges

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   网络解算迭代回路
    def get_circuits(self, weights=None, filterEdges=[], filterVirtual=True) -> list:
        """
        edges           :   通风网络拓扑关系, [{'id':str,'s':str,'t':str},...]
        weights         :   [float,...], 风路权重(>=0)列表, 次序与edges一致
        filterEdges     :   [str,...], 不参与迭代的边，如固定风量边
        fileterVirtual  :   True/False, 过滤虚拟风路   
        """
        return get_circuits(self.edges,weights=weights,filterEdges=filterEdges,filterVirtual=filterVirtual)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€



    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取eid串联边
    def get_edges_series_eid(self, eid) -> list:
        g = GraphJ(edges=self.edges)
        return g.get_edges_series_eid(eid=eid)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取vid串联边       vid端点可能存在多条
    def get_edges_series_vid(self, vid) -> list:
        g = GraphJ(edges=self.edges)
        return g.get_edges_series_vid(vid=vid)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取全风网串联边
    def get_edges_series_all(self):
        g = GraphJ(edges=self.edges)
        return g.get_edges_series_all()
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   串联简化
    def graph_series_simplification(self) -> list:
        g = GraphJ(edges=self.edges)
        return g.graph_series_simplification()
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索最大通路、通路数、搜索成功着色标记等  用于判断复杂度及校验igraph
    def get_sumpath_maxh_maxpath_depth_first_traversal_search(
        self,
        edges, 
        source          =   None, 
        to              =   None, 
        giveways        =   []
    ) :
        return _get_sumpath_maxh_maxpath_depth_first_traversal_search_(
            edges,
            source      =   source,
            to          =   to,
            giveways    =   giveways
        )


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   汇点子网-公共分支
    def get_sink_node_subnet_and_common_edge(self, edges) :
        with context_jian_graph(edges) as jG :

            sinkNodes = jG.sinkVIds

            sinkNodeSubnet = dict()

            for sinkv in sinkNodes :
                eids, trake = jG.depth_first_search_traverses_coloring(targetNodes=sinkv)
                sinkNodeSubnet[sinkv] = [jG.me[eid] for eid in eids]

            esink = dict()
            for e in edges : esink[e['id']] = 0

            for sinkv, subnet in sinkNodeSubnet.items() :
                for e in subnet :
                    esink[e['id']] += 1

            commone = [eid for eid,sum in esink.items() if sum>=2]
        return sinkNodeSubnet, commone


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   截流边
    from .get_closure_edges import  get_closure_related_edges   as  _get_closure_related_edges_
    def get_closure_related_edges(self,closure_edges, starts=None, targets=None) :
        pass
    

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索遍历着色
    def depth_first_search_traverses_coloring(
        self, 
        startNodes  =   None,       # None/str/list, 搜索起始点, None全部源点
        targetNodes =   None,       # None/str/list, 搜索目标点, None全部汇点
        avoidEdges  =   None,        # None/str/list, 避让分支
        record      =   False       # 记录搜索轨迹
    ) :
        g = GraphJ(edges=self.edges)
        return g.depth_first_search_traverses_coloring(
            startNodes  =   startNodes,
            targetNodes =   targetNodes,
            avoidEdges  =   avoidEdges,
            record      =   record       # 记录搜索轨迹
        )
    

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
