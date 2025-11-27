#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	有向图类模块
#
###########################################################################################################################

from    .create_igraph      import  create_igraph   as  _create_igraph_
from    .get_paths          import  get_paths       as  _get_paths_

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   最小树-余树-回路
#
def get_mintree_cotree_circuits(edges, weights=None) :

        # 1. 最小树、余树
        g = _create_igraph_(edges=edges)
        mst = g.spanning_tree(weights=weights, return_tree=True)       # return_tree=False，输出格式index
        tree = mst.es['id']
        cotree = list(set(g.es['id'])-set(tree))

        # 2. 最小树创建无向拓扑图
        edgesTree = list(filter(lambda e : e['id'] in tree, edges))
        edgesCotree = list(filter(lambda e : e['id'] in cotree, edges))

        # 3. 求回路
        circuits = list()
        for e in edgesCotree :      # 余支循环
            paths = _get_paths_(edgesTree, source=e['t'], to=e['s'], undirected=True)       # 余支末始点在最小树中找通路
            circuit = [e['id']] + paths[0]
            circuits.append(circuit)
        return tree, cotree, circuits    
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
