#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	创建闭合网络虚拟分支模块
#
###########################################################################################################################

#   利用节点，注意公共源点、汇点情况，风筒与掘进巷道形成公共汇点

from    jl.network.i_graph.graph    import  G
from    cloud.jl.network.get_topology         import  get_topology    as  _get_topology_

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建闭合网络虚拟分支
#
def get_virtual_edges(edges) :
    g = G()
    # top = g.get_topology(edges=edges)
    top = _get_topology_(edges)

    sourceNodes = top['sourceNodes']
    sinkNodes = top['sinkNodes']
    mape = top['mape']
    # print(top)

    # 1. 基点-源点虚拟分支
 
    virtualSourceEdges = []
    virtualSinkEdges = []

    for i, vid in enumerate(sourceNodes) :
        edge = {'id':'virtual_source_%d' % (i+1), 's':'virtual', 't':vid}
        virtualSourceEdges.append(edge)
    # print(virtualSourceEdges)

    for i, vid in enumerate(sinkNodes) :
        edge = {'id':'virtual_sink_%d' % (i+1), 's':vid, 't':'virtual'}
        virtualSinkEdges.append(edge)
    # print(virtualSinkEdges)
    return virtualSourceEdges+virtualSinkEdges, virtualSourceEdges, virtualSinkEdges

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€




