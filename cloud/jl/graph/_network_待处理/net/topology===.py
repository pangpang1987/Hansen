#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	矿井通风网络图类
#
###########################################################################################################################

from    jl.network.j_graph.jian_graph       import  JianGraph
from    jl.network.i_graph.graph   import  G

# from    .circuit        import  set_mintree_cotree_circuits   as  _set_mintree_cotree_circuits_

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   拓扑关系
#
def get_topology(edges, path=False, **mode) -> dict :
    j = JianGraph(edges=edges)

    ss = [e['s'] for e in edges]
    tt = [e['t'] for e in edges]
    ss.extend(tt)
    vs = list(set(ss))

    mape = {**{e['id']:e for e in edges}}

    nodeInEdges     =   {**{v : [] for v in vs}}
    nodeOutEdges    =   {**{v : [] for v in vs}}
    for e in  edges :
        nodeOutEdges[e['s']].append(e)
        nodeInEdges[e['t']].append(e)

    return {
        'esum'          :   len(edges),                     # int, 分支总数
        'vsum'          :   len(vs),                        # int, 节点总数
        'eids'          :   [e['id'] for e in edges],       # list, [eid, ...], 分支id列表
        'vids'          :   vs,                             # list, [vid, ...], 节点id列表
        'sourceNodes'   :   j.get_vs_source_id(edges),           # list, [vid, ...], 源点id列表
        'sinkNodes'     :   j.get_vs_sink_id(edges),             # list, [vid, ...], 汇点id列表
        'sourceEdges'   :   j.get_edges_source(edges),           # list, [eid, ...], 源边id列表
        'sinkEdges'     :   j.get_edges_sink(edges),             # list, [eid, ...], 汇边id列表
        'mape'          :   mape,                           # dict
        'nodeInEdges'   :   nodeInEdges,                    # dict
        'nodeOutEdges'  :   nodeOutEdges                    # dict
    }
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
