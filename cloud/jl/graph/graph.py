#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	图类模块
#
###########################################################################################################################

import  igraph      as      ig
from    .graph_j    import  GraphJ




from    .get_unidirectional_loops   import  get_unidirectional_loops    as  _get_unidirectional_loops_


import  itertools
from    typing  import  Union, List

#   1. 仅设置通风网络常用的函数, 非常用函数直接调用GraphI, GraphJ

# from    .graph_i            import  GraphI
# from    .graph_j            import  GraphJ
# from    .i.get_circuits     import  get_circuits


from pprint import  pprint

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   图类
#
class G :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   初始化函数
    def __init__(self, edges=None, type='dg'):
        self.edges          =   edges
        if type == 'dg':
            self.dg = self.create_igraph(edges=self.edges,directed=True)
        elif type =='jg':
            self.jg = self.create_jgraph(edges=self.edges)
        elif type == 'all':
            self.dg = self.create_igraph(edges=self.edges,directed=True)
            self.jg = self.create_jgraph(edges=self.edges)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   创建igraph
    def create_igraph(self, edges, directed=True) -> tuple:
        # 1. 创建有向图igraph
        dg = ig.Graph(directed=directed)
        # 2. 实例化iGraph
        # （1）提取节点设置节点索引映射
        ss = list(map(lambda e: e["s"], edges))
        ts = list(map(lambda e: e["t"], edges))
        vs = list(set(ss + ts))
        # （2）添加节点, 设置节点索引映射
        dg.add_vertices(len(vs))  # 添加 5 个节点
        dg.vs["id"] = vs  # 为节点设置id属性
        vid_to_index = {**{v["id"] : v.index  for v in dg.vs}}
        index_to_vid = {**{v.index : v['id'] for v in dg.vs}}
        # （3）添加边
        dg.add_edges([(vid_to_index[e["s"]], vid_to_index[e["t"]]) for e in edges])  # 添加边
        dg.es["id"] = [e["id"] for e in edges]  # 为边设置权重属性
        eid_to_index = {**{e["id"] : e.index  for e in dg.es}}
        index_to_eid = {**{e.index : e['id'] for e in dg.es}}
        dg['vid_to_index'] = vid_to_index
        dg['index_to_vid'] = index_to_vid
        dg['eid_to_index'] = eid_to_index
        dg['index_to_eid'] = index_to_eid
        return dg
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   判断连通图
    def is_connected(self) -> bool :
        return self.dg.is_connected(mode='weak')
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
        components = self.dg.components(mode='weak')      # 计算连通分量
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
        return self.dg.is_dag() # True, 无单向回路,无单节点环；False, 有单向回路或单节点环


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取自环（单节点环）
    def get_self_loops(self) -> list :
        loops = [e["id"] for e in self.edges if e['s']==e['t']]
        return loops

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取单向回路
    #   返回构成单向回路的分支id，这些分支可能构成多个单向回路
    def get_unidirectional_loops(self) -> list:
        return _get_unidirectional_loops_(self.dg)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取源点
    def get_virtices_source(self) -> list :
        return [v['id'] for v in self.dg.vs if all([self.dg.degree(v, mode='in')==0, self.dg.degree(v, mode='all')==1])]
    #----------------
    def get_virtices_source_index(self) -> list :
        return [v.index for v in self.dg.vs if all([self.dg.degree(v, mode='in')==0, self.dg.degree(v, mode='all')==1])]
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取汇点
    def get_virtices_sink(self) -> list :
        return [v['id'] for v in self.dg.vs if all([self.dg.degree(v, mode='out')==0, self.dg.degree(v, mode='all')==1])]    
    #---------------     
    def get_virtices_sink_index(self) -> list :
        return [v.index for v in self.dg.vs if all([self.dg.degree(v, mode='out')==0, self.dg.degree(v, mode='all')==1])]  
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取源边
    def get_edges_source(self) -> list :
        return [e['id'] for e in self.dg.es if all([self.dg.degree(e.source, mode='in')==0,self.dg.degree(e.source, mode='all')==1])]

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取汇边
    def get_edges_sink(self) -> list :
        return [e['id'] for e in self.dg.es if all([self.dg.degree(e.target, mode='out')==0, self.dg.degree(e.target, mode='all')==1])]

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   拓扑检查
    def topology_check(self) :
        return {
            'sum_edges'             :   len(self.dg.es),                        # int, 分支总数
            'sum_virtices'          :   len(self.dg.vs),                        # int, 节点总数
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
        print('get_short_path==================')
        if   isinstance(starts, str)    :   starts = [self.dg['vid_to_index'][starts]]
        elif isinstance(starts, list)   :   starts = [self.dg['vid_to_index'][vid] for vid in starts]
        elif starts is None             :   starts = self.get_virtices_source_index()
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")


        if   isinstance(targets, str)   :   targets = [self.dg['vid_to_index'][targets]]
        elif isinstance(targets, list)  :   targets = [self.dg['vid_to_index'][vid] for vid in targets]
        elif targets is None            :   targets = self.get_virtices_sink_index()
        else                            :   raise TypeError("The input must be a string, a list of strings, or None.")
        
        paths = []
        for source, to in itertools.product(starts, targets):
            path = self.dg.get_shortest_paths(
                source=source,             # 源顶点索引, source,必需,类型：整数; 计算路径的起始顶点索引（从 0 开始）
                to          =   to,            # 目标顶点（默认为所有顶点）
                mode        =   mode,         # 遍历方向："out", "in", "all"
                weights     =   weights,       # 边的权重列表
                output      =   output,     # 输出格式："vpath", "epath", "both"
                algorithm   =   algorithm    # 算法选择（通常自动）
            )[0]
            if path :
                path = [self.dg['index_to_eid'][index] for index in path]
                paths.append({(self.dg['index_to_vid'][source],self.dg['index_to_vid'][to]):path})
        return paths
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   求最小树
    def get_min_tree(self, weights=None) -> list:
        """
        weights:边的权重列表或边属性名称（字符串）。
        若指定权重，函数返回最小生成树（默认按权重升序）；若设为 None(默认），则按边的数量生成树（等价于权重全为 1 的最小生成树）。
        若要生成最大生成树，可传入权重的负值（如 -weights）。
        return_tree:布尔值，默认 True。
        若为 True,返回生成树的 igraph.Graph 对象；
        若为 False,返回生成树的边索引列表（原 graph 中边的索引）。
        """
        graph_tree = self.dg.spanning_tree(weights=weights, return_tree=True)
        return [e for e in self.edges if e['id'] in graph_tree.es['id']]
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   求回路
    def get_circuits(self, weights=None):
        # 1. 用全部分支创建igraph，求最小树
        min_tree_g = self.dg.spanning_tree(weights=weights, return_tree=True)
        min_tree= [e for e in self.edges if e['id'] in min_tree_g.es['id']]
    
        # 2. 用树支重新定义igraph
        # 要重新定义igraph.graph，因为min_tree索引已经发生变化, 与self.dg不一样
        graph_tree = self.create_igraph(min_tree)   # igraph
        cotree = [e for e in self.edges if e['id'] not in min_tree_g.es['id']]
        cs = []
        for e in cotree:
            c = [e['id']]
            path = graph_tree.get_shortest_paths(               # 不能用self
                graph_tree['vid_to_index'][e['t']],             # 源顶点索引
                to=graph_tree['vid_to_index'][e['s']],            # 目标顶点（默认为所有顶点）
                mode="all",         # 遍历方向："out", "in", "all"
                weights=None,       # 边的权重列表, 因最小树中路径是唯一的，所以求最短路可不用权重
                output="epath",     # 输出格式："vpath", "epath", "both"
                algorithm="auto"    # 算法选择（通常自动）
            )[0]
            c += [graph_tree['index_to_eid'][e_index] for e_index in path]
            cs.append(c)
        return cs
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#------------------------
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   创建jGraph
    def create_jgraph(self, edges):
        return GraphJ(edges=edges)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   深度优先搜索遍历着色
    def depth_first_search_traverses_coloring(
        self, 
        start: Union[str, List[str], None]  =   None,       # None/str/list, 搜索起始点, None全部源点
        to: Union[str, List[str], None] =   None,       # None/str/list, 搜索目标点, None全部汇点
        avoidEdges: Union[str, List[str], None]  =   None,        # None/str/list, 避让分支
        record      =   False       # 记录搜索轨迹
    ) :
        # g = GraphJ(edges=self.edges)
        return self.jg.depth_first_search_traverses_coloring(
            start  =   start,
            to =   to,
            avoidEdges  =   avoidEdges,
            record      =   record       # 记录搜索轨迹
        )
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
         
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   正向搜索串联边
    #   从eid开始正向搜索并返回串联边, eid入边部分缺失
    def find_edges_series_forward_eid(self, eid0) :
        return self.jg.find_edges_series_forward_eid(eid0=eid0)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   反向搜索串联边, eid出边部分缺失
    def find_edges_series_reverse_eid(self, eid0) :
        return self.jg.find_edges_series_reverse_eid(eid0=eid0)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取eid串联边
    def get_edges_series_eid(self, eid0) -> list:
        return self.jg.get_edges_series_eid(eid0=eid0)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取vid串联边       vid端点可能存在多条
    def get_edges_series_vid(self, vid) -> list:
        return self.jg.get_edges_series_vid(vid=vid)
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   提取全风网所有串联边
    def get_edges_series_all(self):
        return self.jg.get_edges_series_all()
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
    
   

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
