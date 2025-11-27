#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	有向图类
#
###########################################################################################################################

from    .create_igraph      import  create_igraph

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   同向串联分支
#
def get_series_edge(edges) :

    g = create_igraph(edges=edges)

    seriesv = [v.index for v in g.vs if g.degree(v.index,mode='in')==1 \
                   and g.degree(v.index,mode='out')==1]                         # [index, ...], 串联节点索引, 入度=出度=1
    vcolor = []                                                                 # [index, ...], 着色节点
    seriess = []                                                                # [[eid1,eid2, ...], ...], 串联分支id
    def _find_in_(v0,series) :                                                  # 递归寻找入边
        if v0 in seriesv :
            ei = g.incident(v0, mode='in')
            vcolor.append(v0)
            series.insert(0,ei[0])
            vi = g.es[ei[0]].source
            _find_in_(vi,series)
    def _find_out_(v0,series) :
        if v0 in seriesv :
            eo = g.incident(v0, mode='out')
            vcolor.append(v0)
            series.append(eo[0])
            vo = g.es[eo[0]].target
            _find_out_(vo,series)

    while len(seriesv) :
        v0 = seriesv.pop(0)
        if v0 in vcolor : continue
        series = []                                         # 串联节点串联分支集合
        vcolor.append(v0)
        ei = g.incident(v0, mode='in')
        eo = g.incident(v0, mode='out')
        series.insert(0,ei[0])
        series.append(eo[-1])
        vi = g.es[ei[0]].source
        vo = g.es[eo[0]].target
        vcolor.extend([vi,vo])
        _find_in_(vi,series)
        _find_out_(vo,series)
        series_ = [g.es[i]['id'] for i in series]           # index转id
        seriess.append(series_)

    return seriess
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   无向串联分支    用于删除掘进巷道    与get_series_edge不同在于只要单链接度=2即可
#
def get_series_edge_undirected(edges, eids) :
    # eids 搜索基点id

    g = create_igraph(edges=edges)

    seriess = []

    # 找与v0节点连接，但不是e的分支
    def _find_(v0, eindex, series) :                                                  # 递归寻找入边
        # v0 - 当前节点
        # e - 当前分支
        if g.degree(v0,mode='all')==2 :
            ei = g.incident(v0, mode='in')[0]          # 分支类对象
            eo = g.incident(v0, mode='out')[0]
            if ei != eindex : e=g.es[ei]
            if eo != eindex : e=g.es[eo]
            series.append(e['id'])
            if e.source != v0 :
                _find_(e.source, e.index, series)
            if e.target != v0 :
                _find_(e.source, e.index, series)
        else :
            return

    for eid in eids :
        series = [eid]
        eindex = g.es.select(id=eid)[0].index
        v1index = g.es[eindex].source
        v2index = g.es[eindex].target

        _find_(v1index, eindex, series)
        _find_(v2index, eindex, series)

        seriess.append(series)
    
    # print(seriess)
    return seriess
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

