#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	有向图类（无向图通路、最短路单独设置    注意索引的随机性）
#
###########################################################################################################################

from    cloud.jl.network.get_topology                 import  (
        get_topology                as      _get_topology_,
        is_connected                as      _is_connected_,
        get_connected_graph         as      _get_connected_graph_,
        is_dag                      as      _is_dag_,
        get_loops                   as      _get_loops_,
        get_node_in_edge            as      _get_node_in_edge_,
        get_node_out_edge           as      _get_node_out_edge_
)

from    .g.mintree_cotree_circuits  import  get_mintree_cotree_circuits     as  _get_mintree_cotree_circuits_

from    .g.get_paths                import  get_paths                       as  _get_paths_

from    .g.series                   import  get_series_edge                 as  _get_series_edge_
from    .g.shortest_paths           import  get_shortest_paths              as  _get_shortest_paths_
from    .g.longest_paths            import  (
        get_longest_paths           as      _get_longest_paths_,
        get_longest_path            as      _get_longest_path_
)
from    .g.unidirectional_loop      import  get_unidirectional_loops        as  _get_unidirectional_loops_



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   有向图拓扑类
#
class G() :
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self) : pass                       # 默认有向图

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   主要拓扑关系
    # def get_topology(self, edges) -> dict : return _get_topology_(edges)
    
    def is_connected(self, edges) -> bool : return _is_connected_(edges=edges)

    def get_connected_graph(self,edges) -> list : return _get_connected_graph_(edges)
    
    # 是否检查有向无环图 directed acyclic graph, 
    # 非连通图是DAG, 含单节点环不是DAG、单向回路不是DAG
    def is_dag(self, edges) -> bool : return _is_dag_(edges)
    
    def get_loops(self, edges) -> list : return _get_loops_(edges)
    
    def get_node_in_edge(self, edges) -> dict : return _get_node_in_edge_(edges=edges)
    
    def get_node_out_edge(self, edges) -> dict : return _get_node_out_edge_(edges=edges)


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   同向串联
    def get_series_edge(self, edges) -> list : return _get_series_edge_(edges)    # [[eid1, ...],[eid2, ...], ...]




    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   根据节点id提取节点索引
    def _get_index_vertex(self, vid) : return self.g.vs.select(id=vid)[0].index

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    def _get_id_vertex(self, index) :
        return self.g.vs[index]["id"]

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    def _get_index_edge(self, eid) :
        return self.g.es.select(id=eid)[0].index

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    def _get_id_edge(self, index) :
        return self.g.es[index]['id']
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取节点关联边
    def get_edges_vertex_incident(self, vid, mode='all') :
        # model = 'in', 'out', 'all'
        idx = self.g.vs.select(id=vid)[0].index
        edges = [self.g.es[i]["id"] for i in self.g.incident(idx, mode=mode)]
        return edges

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   节点关联度
    def get_degree(self, vid, mode='all') :
        idx = self.g.vs.select(id=vid)[0].index
        degree = self.g.degree(idx,mode=mode, loops=False)
        return degree


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   通路    数量受限    并联只解析一条
    def get_paths(self, edges,source=None, to=None,cutoff=-1, mode='out', undirected=False) :
        try :
            return _get_paths_(edges, source=source, to=to, cutoff=cutoff,mode=mode,undirected=undirected)
        
        except Exception as e :      # 2. 未知异常中断, 内存超限，
            return []
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   有向图（默认无向）点对最短路
    def get_shortest_paths(self, edges, source=None, to=None, weights=None, undirected=True) ->list :
        return _get_shortest_paths_(edges, source=source, to=to, weights=weights, undirected=undirected)
  
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   每个点对最长路  权重为负 必须有向图, 无向图出错 无权重默认1结果是最短路
    def get_longest_paths(self, edges, source=None, to=None, weights=None) -> list :
        return _get_longest_paths_(edges=edges, source=source, to=to, weights=weights)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   所有点对最长路
    def get_longest_path(self, edges, source=None, to=None, weights=None) -> list :
        return _get_longest_path_(edges,source=source, to=to, weights=weights)
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   最小树-余树-回路   
    def get_mintree_cotree_circuits(self, edges, weights=None) -> tuple :
        return _get_mintree_cotree_circuits_(edges, weights=weights)   # ([eid,...],[eid,...],[[eid1,...],...])



    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   单向回路
    def get_unidirectional_loops(self, edges) : return _get_unidirectional_loops_(edges=edges)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


