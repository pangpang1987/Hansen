#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#   矿井通风网络模块
#
###########################################################################################################################

#   主要是网络解算与调节
#   使用类，尽管有实例化开销，但是迭代数据更大

from jl.graph.graph_jian                 import  G

from pprint                             import  pprint
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#	矿井通风网络类
#
class VN :

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #	构造函数    上一级函数接受dataNS
    def __init__(self,*args,**kwargs) :
        """
        roads       :   list, 风路列表,[{"id":str, "s":str, "t":str, "weight":float}]
        """
        pass

 
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #   风量初始化
    #   回路
    #   最小树
    #   迭代
    #   更新风阻，权重，多次迭代了，调节、反演均可用
    #   
 
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   网络解算迭代回路
    #   风量初始化已经完成
    #   过滤固定风量
    #   算完回路再过滤虚拟风路
    def get_circuits(self, roads, filterEdges=[]) :
        """
        weights     :   list, 风路权重(>=0)列表, 次序与edges一致
        fixedEdges  :   list, 不参与迭代的过滤风路列表        
        """
        
        mape = {road['id']:road for road in roads}

        # 1. 创建有向图
        g = G(roads)

        # 2. 创建闭合网络虚拟风路（先设置虚拟分支, 再过滤）

        # 2.1 提取源汇节点
        source_nodes    =   g.get_virtices_source()
        sink_nodes      =   g.get_virtices_sink()

        # # 2.2 创建虚拟风路
        # virtual_es_source   = []
        # virtual_es_sink     = []
        # virtual_eids        = []

        for i, vid in enumerate(source_nodes) :
            eid = 'virtual_source_%d' % (i+1)
            road = {'id':eid, 's':'virtual', 't':vid, 'weight':.0}
            mape[eid] = road
            roads.append(road)

        for i, vid in enumerate(sink_nodes) :
            eid = 'virtual_sink_%d' % (i+1)
            road = {'id':eid, 's':vid, 't':'virtual', 'weight':.0}
            mape[eid] = road
            roads.append(road)

        # # 3. 设置虚拟风路权重=0
        # virtual_weights = [0] * len(virtual_es_source + virtual_es_sink)

        # 3. 过滤不参与迭代的固定风路（后删除，风筒+掘进巷道，防止删除风筒，掘进道等串联道变风井, 由mine处理）
        roads = list(filter(lambda e : e["id"] not in filterEdges, roads))
            
        # 4. 提取过滤后分支权重
        weights = [e['weight'] for e in roads]

        g = G(roads)

        tree, cotree, circuits  = g.get_tree_cotree_circuits(weights=weights)

        # 分支方向
        # 7.5 过滤回路虚拟分支（热动力模块毋需删除）
        cs = list()
        for circuit in circuits :
            v0 = mape[circuit[0]]['s']
            c = list()
            for eid in circuit :
                e = mape[eid]
                if e['s'] == v0 : 
                    d = 1
                    v0 = e['t']
                else :
                    d = -1
                    v0 = e['s']
                c.append({'eid':eid,'d':d})
            cs.append(c)
        return cs   


    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   与掘进头串联的巷道
    def get_roadways_series_head(self, roads, heads) -> list:
        g = G(edges=roads)
        seriess = []
        for eid in heads :
            seriess.append(g.find_edges_series_reverse(eid))
        return seriess




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   创建闭合网络虚拟分支
    # def create_virtual_edges111(self) :
    #     # 1. 基点-源点虚拟分支
 
    #     virtualSourceEdges = []
    #     virtualSinkEdges = []

    #     for i, vid in enumerate(self.sourceVIds) :
    #         edge = {'id':'virtual_source_%d' % (i+1), 's':'virtual', 't':vid}
    #         virtualSourceEdges.append(edge)
    #     # print(virtualSourceEdges)

    #     for i, vid in enumerate(self.sinkVIds) :
    #         edge = {'id':'virtual_sink_%d' % (i+1), 's':vid, 't':'virtual'}
    #         virtualSinkEdges.append(edge)
    #     # print(virtualSinkEdges)
    #     return virtualSourceEdges+virtualSinkEdges, virtualSourceEdges, virtualSinkEdges

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   独立回路
    #   说明:
    #   （1）可以是非连通图
    #   （2）可以有并联分支
    #   （3）可以有单向回路
    #   （4）分支权重值必须 >= 0
    #   （5）可以无权重
    # def get_circuits(self, weights=None) :
    #     with context_i_graph_algorithm(self.edges) as iGraph :
    #         minTree, coTree, circuits = iGraph.get_mintree_cotree_circuits(weights=weights)

    #     cs_ = list()
    #     for circuit in circuits :
    #         v0 = self.mape[circuit[0]].s.id
    #         c_ = list()
    #         for eid in circuit :
    #             eobj = self.mape[eid]
    #             if eobj.s.id == v0 : 
    #                 d = 1
    #                 v0 = eobj.t.id
    #             else :
    #                 d = -1
    #                 v0 = eobj.s.id
    #             c_.append({'eid':eid,'d':d})
    #         cs_.append(c_)
    #     return cs_, circuits, minTree, coTree