#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	回路模块
#
###########################################################################################################################

#   独立回路说明:
#   （1）可以是非连通图
#   （2）可以有并联分支
#   （3）可以有单向回路
#   （4）分支权重值必须 > 0
#   （5）可以无权重

from    jl.graph.graph      import  G
import  uuid


from pprint import  pprint

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   求迭代回路
#
def get_circuits(roads, filterVirtual=True) :
    """
    roads           :   通风网络拓扑关系, [{'id':str,'s':str,'t':str},...]
    # weights         :   [float,...], 风路权重(>=0)列表, 次序与edges一致
    # filterEdges     :   [str,...], 不参与迭代的边，如固定风量边
    fileterVirtual  :   True/False, 过滤虚拟风路   
    """

    # 1. 创建虚拟风路（需要创建虚拟风量）
    virtuals, virtual_eid = create_virtual_edge(roads)
    # virtuals - 全部虚拟风路
    # virtual_eid - 虚拟风路id

    # 2. 过滤不参与迭代的固定风路（注意过滤不能在1.1之前, 防止出现删除固定风量巷道变风井增加虚拟风路问题）
    # 2.1 固定id
    fixeds = [e['id'] for e in roads if e.get('fixed',None)]
    # 2.2 过滤风路
    roads_ = list(filter(lambda e : e["id"] not in fixeds, roads))
    # 2.2 过滤风路权重（注意次序）
    weights = [e['weight'] for e in roads_]

    # 3. 求回路
    # 3.1 形成闭合通风网络风路
    edges= roads_ + virtuals            # 过滤后的风路+虚拟风量
    weights += len(virtuals)*[.0]       # 虚拟风路权重为0，也可以是无穷小
    g = G(edges=edges)
    cs = g.get_circuits(weights=weights)    # [[eid1,eid2,...],...], eid1为余支
    # print("cs============",cs)
    # 5. 回路加方向
    mape = {**{e['id']:e for e in edges}}
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

    # 6. 过滤虚拟风路
    if filterVirtual :
        cs_ = []
        for c in circuits:
            c_ = list(filter(lambda e : e["eid"] not in virtual_eid, c))
            # print('------',c_)
            cs_.append(c_)
    # pprint(cs_)
    return cs_
# [[{'d': 1, 'eid': 'e4'},
#   {'d': -1, 'eid': 'e9'},
#   {'d': -1, 'eid': 'e7'},
#   {'d': -1, 'eid': 'e5'}],
#  [{'d': 1, 'eid': 'e6'},
#   {'d': -1, 'eid': 'e5'},
#   {'d': -1, 'eid': 'e2'},
#   {'d': 1, 'eid': 'e3'}],
#  [{'d': 1, 'eid': 'e8'},
#   {'d': -1, 'eid': 'e7'},
#   {'d': -1, 'eid': 'e5'},
#   {'d': -1, 'eid': 'e2'},
#   {'d': 1, 'eid': 'e3'}]]
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   闭合网络虚拟分支    放在vn
def create_virtual_edge(roads) -> tuple :
    # 1. 创建虚拟节点（将来改为基点）
    virtual_id = str(uuid.uuid4())

    # 2. 提取源汇点
    g = G(edges=roads)

    sourceNodes = g.get_virtices_source()        # 要使用节点，不要使用分支，防止共用节点情形
    sinkNodes = g.get_virtices_sink()

    virtualSourceEdges = []
    virtualSinkEdges = []
    virtual_eids = []

    for i, vid in enumerate(sourceNodes) :
        eid = virtual_id + '_in_%d' % (i+1)
        virtual_eids.append(eid)
        edge = {'id':eid, 's':virtual_id, 't':vid}
        virtualSourceEdges.append(edge)

    for i, vid in enumerate(sinkNodes) :
        eid = virtual_id + '_out_%d' % (i+1)
        virtual_eids.append(eid)
        edge = {'id':eid, 's':vid, 't':virtual_id}
        virtualSinkEdges.append(edge)

    return virtualSourceEdges+virtualSinkEdges, virtual_eids
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
