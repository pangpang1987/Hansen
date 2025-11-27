#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	最小树-余树-回路函数模块
#
###########################################################################################################################

#   说明:
#   （1）可以是非连通图
#   （2）可以有并联分支
#   （3）可以有单向回路
#   （4）分支权重值必须 >= 0
#   （5）可以无权重

from    jl.network.i_graph.g.mintree_cotree_circuits    import  get_mintree_cotree_circuits     as  _get_mintree_cotree_circuits_
from    jl.network.j_graph.jian_graph       import  JianGraph

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   设置回路方向
#
def set_mintree_cotree_circuits(edges, weights=None) :
    minTree, coTree, circuits = _get_mintree_cotree_circuits_(edges=edges, weights=weights)

    j = JianGraph(edges)

    cs_ = list()

    for circuit in circuits :
        v0 = j.mape[circuit[0]]['s']
        c_ = list()
        for eid in circuit :
            e = j.mape[eid]
            if e['s'] == v0 : 
                d = 1
                v0 = e['t']
            else :
                d = -1
                v0 = e['s']
            c_.append({'eid':eid,'d':d})
        cs_.append(c_)
    return minTree, coTree, cs_
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

