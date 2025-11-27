#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	回路模块
#
###########################################################################################################################

    #   独立回路
    #   说明:
    #   （1）可以是非连通图
    #   （2）可以有并联分支
    #   （3）可以有单向回路
    #   （4）分支权重值必须 >= 0
    #   （5）可以无权重

# from    .create_igraph  import  create_igraph
from    jl.graph.graph_i   import  GraphI

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求迭代回路
#
def get_circuits(edges, weights=None, filterEdges=[], filterVirtual=True) :
    """
    edges           :   通风网络拓扑关系, [{'id':str,'s':str,'t':str},...]
    weights         :   [float,...], 风路权重(>=0)列表, 次序与edges一致
    filterEdges     :   [str,...], 不参与迭代的边，如固定风量边
    fileterVirtual  :   True/False, 过滤虚拟风路   
    """

    # 1. 创建虚拟风路

    # 1.1 创建edges对应的GraphI对象
    g1 = GraphI(edges=edges)
    

    # 1.2 创建虚拟风路
    virtuals, virtual_eid = create_virtual_edge(g1) 

    # 2. 过滤不参与迭代的固定风路、（注意删除固定风量巷道变风井问题）

    # 2.1 过滤
    edges_1 = list(filter(lambda e : e["id"] not in filterEdges, edges))

    # 2.2 过滤风路权重（注意次序）
    weights = [e['weight'] for e in edges_1]

    # 3. 求最小树、余树

    # 3.1 形成闭合通风网络风路
    edges_2= edges_1+virtuals
    weights += len(virtuals)*[.0]       # 虚拟风路权重为0，也可以是无穷小

    print(edges_2)
    print(weights)
    input()
    # 3.2 求最小树
    g2, vid_to_index, index_to_vid, eid_to_index, index_to_eid  = create_igraph(edges_2)
    tree_index = g2.spanning_tree(weights=weights,return_tree=False)

    # 3.3 索引树转id树，求余树
    mintree_eid = []
    cotree_eid = []
    for e in g2.es:
        if e.index in tree_index:
            mintree_eid.append(e['id'])
        else:
            cotree_eid.append(e['id'])

    # 4. 求回路

    # 4.1 创建最小树对应的igraph
    edge_3 = list(filter(lambda e : e["id"] not in cotree_eid, edges_2))
    iG3, vid_to_index, index_to_vid, eid_to_index, index_to_eid  = create_igraph(edge_3)

    edges_cotree = [e for e in edges if e['id'] in cotree_eid]

    # 4.2 最短路法求回路，余支模节点为最短路起始点，始节点为搜索目标点
    cs = []
    for e in edges_cotree:
        c = [e['id']]
        path = iG3.get_shortest_paths(
            vid_to_index[e['t']],             # 源顶点索引
            to=vid_to_index[e['s']],            # 目标顶点（默认为所有顶点）
            mode="all",         # 遍历方向："out", "in", "all"
            weights=None,       # 边的权重列表
            output="epath",     # 输出格式："vpath", "epath", "both"
            algorithm="auto"    # 算法选择（通常自动）
        )[0]
        c += [index_to_eid[e_index] for e_index in path]
        cs.append(c)
    
    # 5. 回路加方向
    mape = {**{e['id']:e for e in edges+virtuals}}
    circuits = []
    for circuit in cs :
        v0 = mape[circuit[0]]['s']
        c_ = list()
        for eid in circuit :
            e = mape[eid]
            if e['s'] == v0 : 
                d = 1
                v0 = e['t']
            else :
                d = -1
                v0 = e['s']
            c_.append({'eid':eid,'d':d})
        circuits.append(c_)

    # pprint(circuits)

    # 6. 过滤虚拟风路
    if filterVirtual :
        cs_ = []
        for c in circuits:
            c_ = list(filter(lambda e : e["eid"] not in virtual_eid, c))
            # print('------',c_)
            cs_.append(c_)
    return cs_
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   闭合网络虚拟分支    放在vn
def create_virtual_edge(g) -> tuple :

    # 1. 提取源汇点
    sourceNodes = g.get_virtices_source()        # 要使用节点，不要使用分支，防止共用节点情形
    sinkNodes = g.get_virtices_sink()

    virtualSourceEdges = []
    virtualSinkEdges = []
    virtual_eids = []

    for i, vid in enumerate(sourceNodes) :
        eid = 'virtual_source_%d' % (i+1)
        virtual_eids.append(eid)

        edge = {'id':eid, 's':'virtual', 't':vid}
        virtualSourceEdges.append(edge)

    for i, vid in enumerate(sinkNodes) :
        eid = 'virtual_sink_%d' % (i+1)
        virtual_eids.append(eid)
        edge = {'id':eid, 's':vid, 't':'virtual'}
        virtualSinkEdges.append(edge)

    return virtualSourceEdges+virtualSinkEdges, virtual_eids
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
