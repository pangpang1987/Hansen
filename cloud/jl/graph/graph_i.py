#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph派生类模块
#
###########################################################################################################################

from    typing                          import  Union, List
import  itertools

from    .i.create_igraph                import  create_igraph
from    .get_unidirectional_loops     import  get_unidirectional_loops
from    .i.get_shortest_paths           import  get_shortest_paths

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   igraph.Graph派生类
#
class GraphI:

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   初始化函数
    def __init__(self, *args, edges=None, directed=True, **kwargs):

        # 2. 存储edges
        self.edges = edges
        print(self.edges)
        input()
        self.ig, self.vid_to_index, self.index_to_vid, self.eid_to_index, self.index_to_eid = create_igraph(self.edges,directed=True)

        self.mape = {**{e['id']:e for e in self.edges}}

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   判断连通图
    def is_connected(self) -> bool :
        return self.ig.is_connected(mode='weak')
    #   参数说明
    #   is_connected(mode="weak") 接受一个 mode 参数，适用于有向图：
    #   无向图：忽略 mode 参数，直接判断是否存在从任意顶点到其他顶点的路径。
    #   有向图：
    #   mode="weak"（默认）：检查弱连通性（将有向边视为无向边后是否连通）。
    #   mode="strong"：检查强连通性（任意两顶点间存在双向路径）。
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取连通图
    def get_connected_graphs(self) -> list:
        components = self.ig.components(mode='weak')      # 计算连通分量
        connectedBlocks = []
        for subg in components.subgraphs():         # 连通分量转换成连通子图并遍历循环
            # 子图的index与原图已经不一致了!
            # 这里只能使用id属性进行返回!
            connectedBlocks.append([e['id'] for e in subg.es])
        return connectedBlocks
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   判断有向图是否为有向无环图（Directed Acyclic Graph, DAG）
    def is_dag(self) :
        return self.ig.is_dag() # True, 无单向回路,无单节点环；False, 有单向回路或单节点环


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取自环（单节点环）
    def get_self_loops(self) -> list :
        loops = [e["id"] for e in self.edges if e['s']==e['t']]
        return loops

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取单向回路
    def get_unidirectional_loops(self) -> list:
        return get_unidirectional_loops(self.edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取源点
    def get_virtices_source(self) -> list :
        return [v['id'] for v in self.ig.vs if self.ig.degree(v, mode='in')==0 \
                and self.ig.degree(v, mode='all')==1]
    
    def get_virtices_source_index(self) -> list :
        return [v.index for v in self.ig.vs if self.ig.degree(v, mode='in')==0 \
                and self.ig.degree(v, mode='all')==1]
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取汇点
    def get_virtices_sink(self) -> list :
        return [v['id'] for v in self.ig.vs if self.ig.degree(v, mode='out')==0 \
                and self.ig.degree(v, mode='all')==1]    
              
    def get_virtices_sink_index(self) -> list :
        return [v.index for v in self.ig.vs if self.ig.degree(v, mode='out')==0 \
                and self.ig.degree(v, mode='all')==1]  
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取源边
    def get_edges_source(self) -> list :
        return [e['id'] for e in self.ig.es if self.ig.degree(e.source, mode='in')==0 \
                and self.ig.degree(e.source, mode='all')==1]

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    def get_edges_sink(self) -> list :
        return [e['id'] for e in self.ig.es if self.ig.degree(e.target, mode='out')==0 \
                and self.ig.degree(e.target, mode='all')==1]

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   拓扑检查
    def topology_check(self) :
        return {
            'sum_edges'             :   len(self.ig.es),                        # int, 分支总数
            'sum_virtices'          :   len(self.ig.vs),                        # int, 节点总数
            'is_connected'          :   self.is_connected(),
            'sum_connected_graphs'  :   len(self.get_connected_graphs()),
            'is_dag'                :   self.is_dag(),
            'self_loops'            :   self.get_self_loops(),
            'source_nodes'          :   self.get_virtices_source(),
            'sink_nodes'            :   self.get_virtices_sink(),
            'source_edges'          :   self.get_edges_source(),
            'sink_edges'            :   self.get_edges_sink(),
            'unidirectional_loops'  :   self.get_unidirectional_loops()
        }  

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   求最短路                    负权重不要使用
    def get_shortest_paths(
        self,
        starts  : Union[str, List[str], None]   =   None,       # 搜索起始点, str/[str]/None, None-全部网络源点. 与igraph不同
        targets : Union[str, List[str], None]   =   None,       # 搜索目标点, str/[str]/None, None-全部网络汇点. 与igraph不同
        mode                                    =   "all",      # 遍历方向: "out"-从源顶点出发的路径（有向图的默认设置）; "in"：指向源顶点的路径; "all"：忽略方向（无向图的默认设置）
        weights                                 =   None,       # 边的权重列表,边权重)类型：列表或 None默认：None（所有边权重为 1，即无权图）规则：权重值越大表示路径越长。若需表示实际距离，可用正数；负数权重可能导致未定义行为。
        output                                  =   "epath",    # 输出格式 : "vpath"-返回顶点路径（默认）; "epath"-返回边路径; "both"-同时返回顶点和边路径; 不可达路径返回 None。
        algorithm                               =   "auto"      # 算法选择（通常自动）若存在负权重环，使用 algorithm="bellman-ford"。注意：负权重可能导致无界最短路径（需检查图中无负环）。
    ) -> list:
        
        if   isinstance(starts, str)    :   starts = [self.vid_to_index[starts]]
        elif isinstance(starts, list)   :   starts = [self.vid_to_index[vid] for vid in starts]
        elif starts is None             :   starts = self.get_virtices_source_index()
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")


        if   isinstance(targets, str)   :   targets = [self.vid_to_index[targets]]
        elif isinstance(targets, list)  :   targets = [self.vid_to_index[vid] for vid in targets]
        elif targets is None            :   targets = self.get_virtices_sink_index()
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")
        
        paths = []
        for source, to in itertools.product(starts, targets):
            path = self.ig.get_shortest_paths(
                source,             # 源顶点索引, source,必需,类型：整数; 计算路径的起始顶点索引（从 0 开始）
                to          =   to,            # 目标顶点（默认为所有顶点）
                mode        =   mode,         # 遍历方向："out", "in", "all"
                weights     =   weights,       # 边的权重列表
                output      =   output,     # 输出格式："vpath", "epath", "both"
                algorithm   =   algorithm    # 算法选择（通常自动）
            )[0]
            if path :
                path = [self.index_to_eid[index] for index in path]
                paths.append({(self.index_to_vid[source],self.index_to_vid[to]):path})
        return paths
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #
    # def get_min_path(self,) -> 
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
